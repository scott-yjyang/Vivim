# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
import torch 
from functools import partial
from mamba_ssm import Mamba
import torch.nn.functional as F 
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
# Load model directly
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation



class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)

            x = self.weight[:, None, None] * x + self.bias[:, None, None]

            return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, nf, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, nf, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, nf, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, nf, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class MambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, mlp_ratio=4, drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                # use_fast_path=False,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, C, nf, H, W = x.shape

        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)

        x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))
        x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), nf, H, W))
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out



class mamba_block(nn.Module):
    """
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, backbone, in_chans=1, depths=[2, 2, 2, 2], dims=[64, 128, 320, 512],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        
        self.downsample_layers = backbone.segformer.encoder

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[nn.Sequential(
                MambaLayer(dim=dims[i],drop_path=dp_rates[i])
                    ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices



    def forward_features(self, x):
        outs = []
        bz, nf, nc, h, w = x.shape
        x = x.reshape(bz*nf,x.shape[-3],x.shape[-2],x.shape[-1])

        hs = x
        for idx, x in enumerate(zip(self.downsample_layers.patch_embeddings, self.downsample_layers.block, self.downsample_layers.layer_norm, self.stages)):
            embedding_layer, block_layer, norm_layer, mam_stage = x
            # first, obtain patch embeddings
            # print(hs.shape)
            hs, height, width = embedding_layer(hs)

            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                layer_outputs = blk(hs, height, width, False)
                hs = layer_outputs[0]

            # third, apply layer norm
            # hs = norm_layer(hs)
            # fourth, optionally reshape back to (batch_size, num_channels, height, width)
            hs = hs.reshape(bz*nf, height, width, -1).permute(0, 3, 1, 2).contiguous()
            # print(hs.shape)
            hs = hs.reshape(bz,nf,hs.shape[-3],hs.shape[-2],hs.shape[-1]).transpose(1,2)
            # print(x.size())
            hs = mam_stage(hs)
            # print(x.shape)
            hs = hs.transpose(1,2)
            hs = hs.reshape(bz*nf,hs.shape[-3],hs.shape[-2],hs.shape[-1])

            # print(x_out.shape)
            outs.append(hs)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        # for xx in x:
        #     print(xx.shape)
        return x


class Vivim(nn.Module):
    def __init__(
        self,
        in_chans=3,
        out_chans=1,
        depths=[2, 2, 2, 2],
        feat_size=[64, 128, 320, 512],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=2,
        with_edge=False,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims

        backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        self.encoder = mamba_block(backbone, in_chans, dims=feat_size,
                              )
        self.decoder = backbone.decode_head
        # self.decoder.classifier = nn.Sequential()

        self.out = nn.Conv2d(768, out_chans, kernel_size=1)
        # self.conv_proj = ProjectionHead(dim_in=hidden_size)
        self.with_edge = with_edge
        if with_edge:
            self.edgeocr_cls_head = nn.Conv2d(
                64, 1, kernel_size=1, stride=1, padding=0,
                bias=True)



    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def decode(self, encoder_hidden_states, bz, nf):
    # def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.decoder.linear_c):
            if self.decoder.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
                )

            # unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)
        concat_hidden_states = torch.cat(all_hidden_states[::-1], dim=1)
        hidden_states = self.decoder.linear_fuse(concat_hidden_states)
        hidden_states = self.decoder.batch_norm(hidden_states)
        hidden_states = self.decoder.activation(hidden_states)
        hidden_states = self.decoder.dropout(hidden_states)


        logits = self.out(hidden_states)

        return logits        


    def forward(self, x_in):
        bz, nf, nc, h, w = x_in.shape
        outs = self.encoder(x_in)
        logits = self.decode(outs,bz,nf)
        # print(last_feat.shape)

        # print(logits.shape)
        upsampled_logits = nn.functional.interpolate(
                logits, size=(h,w), mode="bilinear", align_corners=False
            )

        if self.with_edge:
            edge = self.edgeocr_cls_head(outs[0])
            edge = nn.functional.interpolate(
                    edge, size=(h,w), mode="bilinear", align_corners=False
                )
            return upsampled_logits, edge
        else:
            return upsampled_logits
    

