from typing import Optional
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import copy
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm

import pytorch_lightning as pl
import yaml
from easydict import EasyDict
import random
from pytorch_lightning import callbacks
from pytorch_lightning.accelerators import accelerator
from pytorch_lightning.core.hooks import CheckpointHooks
from pytorch_lightning.callbacks import ModelCheckpoint,DeviceStatsMonitor,EarlyStopping,LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace

from data_polyp import get_trainloader,get_testloader
from torch.utils.data import DataLoader
from loss import *
# from models2.refinenet import RefineNet
from torchvision.utils import save_image

output_dir = 'logs'
version_name='Baseline'
logger = TensorBoardLogger(name='vivim_polyp',save_dir = output_dir )
import matplotlib.pyplot as plt
# import tent
import math

from medpy import metric
# from misc import *
import misc2
import torchmetrics
from modeling.vivim import Vivim

from poloy_metrics import *
from modeling.utils import JointEdgeSegLoss
# torch.set_float32_matmul_precision('high')

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()


class CoolSystem(pl.LightningModule):
    
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.params = hparams
        self.epochs = self.params.epochs
        self.save_path='/home/yijun/project/ultra/save_images_polyp2'
        self.data_root=self.params.data_root
        self.initlr = self.params.initlr

        self.train_batchsize = self.params.train_bs
        self.val_batchsize = self.params.val_bs
    
    
        #Train setting
        self.initlr = self.params.initlr #initial learning
        self.weight_decay = self.params.weight_decay #optimizers weight decay
        self.crop_size = self.params.crop_size #random crop size
        self.num_workers = self.params.num_workers
        self.epochs = self.params.epochs
        self.shift_length = self.params.shift_length
        self.val_aug = self.params.val_aug
        self.with_edge = self.params.with_edge

        self.gts = []
        self.preds = []
        
        
        self.nFrames = 5
        self.upscale_factor = 1
        self.data_augmentation = True

        self.criterion = JointEdgeSegLoss(classes=2) if self.with_edge else structure_loss

        self.model = Vivim(with_edge=self.with_edge)


        self.save_hyperparameters()
        

    def configure_optimizers(self):
        # REQUIRED
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.initlr,betas=[0.9,0.999])#,weight_decay=self.weight_decay)
         
        # optimizer = Lion(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.initlr,betas=[0.9,0.99],weight_decay=0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=self.initlr * 0.01)

        return [optimizer], [scheduler]

    # def training_epoch_start(self):
    #     self.scheduler = pipeline.create_SR3scheduler(self.diff_opt['scheduler'], 'train')

    def init_weight(self,ckpt_path=None):
        
        if ckpt_path:
            checkpoint = torch.load(ckpt_path)
            print(checkpoint.keys())
            checkpoint_model = checkpoint
            state_dict = self.model.state_dict()
            # # 1. filter out unnecessary keys
            checkpoint_model = {k: v for k, v in checkpoint_model.items() if k in state_dict.keys()}
            print(checkpoint_model.keys())
            # 2. overwrite entries in the existing state dict
            state_dict.update(checkpoint_model)
            
            self.model.load_state_dict(checkpoint_model, strict=False) 


    def evaluate_one_img(self, pred, gt):


        dice = misc2.dice(pred, gt)
        specificity = misc2.specificity(pred, gt)
        jaccard = misc2.jaccard(pred, gt)
        precision = misc2.precision(pred, gt)
        recall = misc2.recall(pred, gt)
        f_measure = misc2.fscore(pred, gt)

        return dice, specificity, precision, recall, f_measure, jaccard



    def training_step(self, batch, batch_idx):
        self.model.train()

        
        neigbor, target, edge_gt = batch
        # print(edge_gt.shape)
        target = target.cuda()
        #bicubic = bicubic.cuda()
        neigbor = neigbor.cuda()
        bz, nf, nc, h, w = target.shape
        # noisy_images = torch.cat([noisy_images,neigbor_],dim=1)
        # print(neigbor.shape)
        #print("timesteps:",timesteps)#.type())
        if not self.with_edge:
            pred = self.model(neigbor)#, return_dict=False)[0]
            target = target.reshape(bz*nf,nc,h,w)
            loss = self.criterion(pred[self.nFrames//2::self.nFrames], target[self.nFrames//2::self.nFrames])

        else:
            pred,e0 = self.model(neigbor)#, return_dict=False)[0]
            target = target.reshape(bz*nf,nc,h,w)
            edge_gt = edge_gt.reshape(bz*nf,1,h,w)
            loss = self.criterion((pred[self.nFrames//2::self.nFrames], e0[self.nFrames//2::self.nFrames]), (target[self.nFrames//2::self.nFrames], edge_gt[self.nFrames//2::self.nFrames]))

        self.log("train_loss",loss,prog_bar=True)
        # self.log("aux_loss",aux_loss,prog_bar=True)
        return {"loss":loss}


    def on_validation_epoch_end(self):

        self.sm = Smeasure()
        self.em = Emeasure()
        self.mae = MAE()

        dice_lst, specificity_lst, precision_lst, recall_lst, f_measure_lst, jaccard_lst = [], [], [], [], [], []
        Thresholds = np.linspace(1, 0, 256)
        # print(Thresholds)
        for pred, gt in zip(self.preds,self.gts):
            pred = torch.sigmoid(pred)
            # gt = gt.to(int)
            self.sm.step(pred.squeeze(0).squeeze(0).detach().cpu().numpy(),gt.squeeze(0).squeeze(0).detach().cpu().numpy())
            self.em.step(pred.squeeze(0).squeeze(0).detach().cpu().numpy(),gt.squeeze(0).squeeze(0).detach().cpu().numpy())
            self.mae.step(pred.squeeze(0).squeeze(0).detach().cpu().numpy(),gt.squeeze(0).squeeze(0).detach().cpu().numpy())
            gt = (gt>0.5).to(int)
            dice_l, specificity_l, precision_l, recall_l, f_measure_l, jaccard_l = [], [], [], [], [], []
            for j, threshold in enumerate(Thresholds):
                # print(threshold)
                pred_one_hot = (pred>threshold).to(int)
                
                dice, specificity, precision, recall, f_measure, jaccard = self.evaluate_one_img(pred_one_hot.detach().cpu().numpy(), gt.detach().cpu().numpy())
                # print(dice)
                dice_l.append(dice)
                specificity_l.append(specificity)
                precision_l.append(precision)
                recall_l.append(recall)
                f_measure_l.append(f_measure)
                jaccard_l.append(jaccard)
            dice_lst.append(sum(dice_l) / len(dice_l))
            specificity_lst.append(sum(specificity_l) / len(specificity_l))
            precision_lst.append(sum(precision_l) / len(precision_l))
            recall_lst.append(sum(recall_l) / len(recall_l))
            f_measure_lst.append(sum(f_measure_l) / len(f_measure_l))
            jaccard_lst.append(sum(jaccard_l) / len(jaccard_l))


            # print(sum(dice_l) / len(dice_l))

        # mean
        dice = sum(dice_lst) / len(dice_lst)
        acc = sum(specificity_lst) / len(specificity_lst)
        precision = sum(precision_lst) / len(precision_lst)
        recall = sum(recall_lst) / len(recall_lst)
        f_measure = sum(f_measure_lst) / len(f_measure_lst)
        jac = sum(jaccard_lst) / len(jaccard_lst)

        sm = self.sm.get_results()['Smeasure']
        em = self.em.get_results()['meanEm']
        mae = self.mae.get_results()['MAE']

        print(len(self.gts))
        print(len(self.preds))
        
        self.log('Dice',dice)
        self.log('Jaccard',jac)
        self.log('Precision',precision)
        self.log('Recall',recall)
        self.log('Fmeasure',f_measure)
        self.log('specificity',acc)
        self.log('Smeasure',sm)
        self.log('Emeasure',em)
        self.log('MAE',mae)

        self.gts = []
        self.preds = []
        print("Val: Dice {0}, Jaccard {1}, Precision {2}, Recall {3}, Fmeasure {4}, specificity: {5}, Smeasure {6}, Emeasure {7}, MAE: {8}".format(dice,jac,precision,recall,f_measure,acc,sm,em,mae))



    def validation_step(self,batch,batch_idx):
        # torch.set_grad_enabled(True)
        self.model.eval()
        
        neigbor,target = batch
        bz, nf, nc, h, w = neigbor.shape


        # import time
        # start = time.time()
        if not self.with_edge:
            samples = self.model(neigbor)
        else:
            samples,_ = self.model(neigbor)

        samples = samples[self.nFrames//2::self.nFrames]


        filename = "sample_{}.png".format(batch_idx)
        save_image(samples,os.path.join(self.save_path, filename))      
        filename = "target_{}.png".format(batch_idx)
        save_image(target,os.path.join(self.save_path, filename))



        self.preds.append(samples)
        self.gts.append(target)
    
    def train_dataloader(self):
        train_loader = get_trainloader(self.data_root, batchsize=self.train_batchsize, trainsize=self.crop_size)
        return train_loader
    
    def val_dataloader(self):
        val_loader = get_testloader(self.data_root, batchsize=self.val_batchsize, trainsize=self.crop_size)
        return val_loader  


def main():
    RESUME = False
    resume_checkpoint_path = r'/home/yijun/project/ultra/logs/uentm_polyp/version_2/checkpoints/ultra-epoch.ckpt'
    if RESUME == False:
        resume_checkpoint_path =None
    #128: 32-0.0005
    args={
    'epochs': 200,  #datasetsw
    'data_root':'./polyp/',
    
    'train_bs':8,
    'test_bs':1,
    'val_bs':1, 
    'initlr':1e-4,
    'weight_decay':0.01,
    'crop_size':256,
    'num_workers':8,
    'shift_length':32,
    'val_aug':False,
    'with_edge':False,
    'seed': 1234
    }

    torch.manual_seed(args['seed'])
    random.seed(args['seed'])
    np.random.seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.benchmark = True

    hparams = Namespace(**args)

    model = CoolSystem(hparams)

    checkpoint_callback = ModelCheckpoint(
    monitor='Dice',
    #dirpath='/mnt/data/yt/Documents/TSANet-underwater/snapshots',
    filename='ultra-epoch{epoch:02d}-Dice-{Dice:.4f}-Jaccard-{Jaccard:.4f}',
    auto_insert_metric_name=False,   
    every_n_epochs=1,
    save_top_k=1,
    mode = "max",
    save_last=True
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        max_epochs=hparams.epochs,
        accelerator='gpu',
        devices=1,
        precision=32,
        logger=logger,
        strategy="auto",
        enable_progress_bar=True,
        log_every_n_steps=5,
        callbacks = [checkpoint_callback,lr_monitor_callback]
    ) 


    trainer.fit(model,ckpt_path=resume_checkpoint_path)
    # val_path=r'/home/yijun/project/ultra/logs/uentm_polyp/version_60/checkpoints/ultra-epoch.ckpt'
    # trainer.validate(model,ckpt_path=val_path)
    
if __name__ == '__main__':
	#your code
    main()
