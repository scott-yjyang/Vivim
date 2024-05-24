import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance,ImageOps
import csv
import torch
import json
from scipy.ndimage.morphology import distance_transform_edt
#several data augumentation strategies
def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label
def randomCrop(image, label):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)
def randomRotation(image,label):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
    return image,label
def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image
def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))
def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img)  

# dataset for training
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(joints):
            if pt[2] > 0:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or \
                    x >= self.output_res or y >= self.output_res:
                    continue

                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(
                    hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

def augment(img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    
    img_nn = [colorEnhance(j) for j in img_nn]
    if random.random() < 0.5 and flip_h:
        #img_in = ImageOps.flip(img_in)
        img_tar = [ImageOps.flip(j) for j in img_tar]
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            #img_in = ImageOps.mirror(img_in)
            img_tar = [ImageOps.mirror(j) for j in img_tar]
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            #img_in = img_in.rotate(180)
            img_tar = [j.rotate(180) for j in img_tar]
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True
    return img_tar, img_nn

def convert_mask(mask, max_obj):

    # convert mask to one hot encoded
    oh = []
    for k in range(max_obj+1):
        oh.append(mask==k)

    if isinstance(mask, np.ndarray):
        oh = np.stack(oh, axis=0)
    else:
        oh = torch.cat(oh, dim=0).float()

    return oh

class SalObjDataset(data.Dataset):
    def __init__(self, root, trainsize, clip_len=5):
        self.trainsize = trainsize
        image_root = os.path.join(root, "Train")
        vid_list = os.listdir(image_root)
        self.clip_len = clip_len
        self.images = []
        self.gts = []
        for vid in vid_list:
            vid_path = os.path.join(image_root, vid, "Frame")
            if 'Kvasir' not in vid:
                frms = sorted(os.listdir(vid_path),key=lambda x: int(x[:-4]))
            else:
                frms = sorted(os.listdir(vid_path))
            for idx in range(len(frms)):
                clip = []
                for ii in range(-clip_len//2+1, clip_len//2+1):
                    # pick_idx = idx + ii if idx - ii < 0 else idx - ii
                    pick_idx = idx+ii
                    if pick_idx >= len(frms):
                        pick_idx = len(frms) - 1
                    if pick_idx < 0:
                        pick_idx = 0
                    clip.append(os.path.join(vid_path, frms[pick_idx]))
                self.images.append(clip)
                self.gts.append([x.replace("Frame", "GT").replace("jpg","png") for x in clip])

        self.size = len(self.images)
        print(self.size)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        
    
    def __getitem__(self, index):
        images = [self.rgb_loader(x) for x in self.images[index]]
        gt = [self.binary_loader(x) for x in self.gts[index]]
        # gt = self.binary_loader(self.gts[index][0])
        edge=[]
        gt, images = augment(gt, images)
        # gt = gt[self.clip_len//2]
        for i in range(len(gt)):
            gt[i] = np.array(gt[i])
            # gt[i][gt[i]!=0]=255
            gt[i] = Image.fromarray(gt[i])

            gt[i] = self.gt_transform(randomPeper(gt[i]))
            _edgemap = gt[i].clone()
            _edgemap = convert_mask(_edgemap,1)
            _edgemap = _edgemap.numpy()
            # _edgemap = self.mask_to_onehot(_edgemap, 2)
            _edgemap = self.onehot_to_binary_edges(_edgemap, 2, 2)
            _edgemap = torch.from_numpy(_edgemap).float()
            edge.append(_edgemap)

        images = [self.img_transform(x) for x in images]

        
        return torch.stack(images), torch.stack(gt), torch.stack(edge)

    def mask_to_onehot(self, mask, num_classes):
        _mask = [mask == (i) for i in range(num_classes)]
        return np.array(_mask).astype(np.uint8)
    
    def onehot_to_binary_edges(self, mask, radius, num_classes):
        mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
        edgemap = np.zeros(mask.shape[1:])
        for i in range(num_classes):
            dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist
        # edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap > 0).astype(np.uint8)
        return edgemap

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


from operator import itemgetter

class SalObjTestDataset(data.Dataset):
    def __init__(self, root, trainsize, clip_len=5):
        self.trainsize = trainsize
        image_root = os.path.join('./polyp/CVC-ClinicDB-612-Test', "Frame")
        vid_list = os.listdir(image_root)
        vid_list = sorted(vid_list,key=lambda x: int(x))
        self.clip_len=clip_len
        self.images = []
        self.gts = []
        # print(vid_list)
        for vid in vid_list:
            vid_path = os.path.join(image_root, vid)
            frms = sorted(os.listdir(vid_path),key=lambda x: int(x[:-4]))
            # print(frms)
            for idx in range(len(frms)):
                clip = []
                for ii in range(-clip_len//2+1, clip_len//2+1):
                    # pick_idx = idx + ii if idx - ii < 0 else idx - ii
                    pick_idx = idx+ii
                    if pick_idx >= len(frms):
                        pick_idx = len(frms) - 1
                    if pick_idx < 0:
                        pick_idx = 0
                    clip.append(os.path.join(vid_path, frms[pick_idx]))
                self.images.append(clip)
                self.gts.append([x.replace("Frame", "GT").replace("jpg","png") for x in clip])

        self.size = len(self.images)
        print(self.size)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        

    
    def __getitem__(self, index):
        images = [self.rgb_loader(x) for x in self.images[index]]
        gt = self.binary_loader(self.gts[index][self.clip_len//2])

        gt = np.array(gt)
        # gt[gt!=0]=255
        gt = Image.fromarray(gt)
        gt = self.gt_transform(gt)

        images = [self.img_transform(x) for x in images]
        
        return torch.stack(images), gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


###############################################################################
#

#dataloader for training
def get_trainloader(image_root, batchsize, trainsize, clip_len=5, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, trainsize, clip_len)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_testloader(image_root, batchsize, trainsize, clip_len=5, shuffle=False, num_workers=12, pin_memory=True):

    dataset = SalObjTestDataset(image_root, trainsize, clip_len)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader
