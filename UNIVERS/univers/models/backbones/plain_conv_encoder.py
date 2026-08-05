# Copyright (c) nnUNet v2

from typing import Union, Type, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16

from mmdet.models.builder import BACKBONES


from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd


@BACKBONES.register_module()
class PlainConvEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 return_skips: bool = False,
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='IN3d', affine=True),
                 act_cfg=dict(type='LeakyReLU', inplace=True),
                 ):

        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        assert len(kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                             "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        stages = []
        for s in range(n_stages):
            stage_modules = []
            conv_stride = strides[s]
            stage_modules.append(
                ConvModule(
                    input_channels, features_per_stage[s], kernel_size=kernel_sizes[s], stride=conv_stride, padding=kernel_sizes[s]//2,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            )
            for _ in range(n_conv_per_stage[s] - 1):
                stage_modules.append(
                    ConvModule(
                        features_per_stage[s], features_per_stage[s], kernel_size=kernel_sizes[s], stride=1, padding=kernel_sizes[s]//2,
                        conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                ) 
            stages.append(nn.Sequential(*stage_modules))
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.return_skips = return_skips

    def forward(self, x):
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]
        
    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)

    def init_weights(self):
        self.initialize(self)

    
class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)