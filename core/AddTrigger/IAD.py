'''
This is the implement of IAD [1]. 
This code is developed based on its official codes (https://github.com/VinAIResearch/input-aware-backdoor-attack-release)

Reference:
[1] Input-Aware Dynamic Backdoor Attack. NeurIPS 2020.
'''

import warnings
warnings.filterwarnings("ignore")
import os
import os.path as osp
import time
from copy import deepcopy
import random
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10

from ..utils import Log


class Normalize:
    """Normalization of images.

    Args:
        dataset_name (str): the name of the dataset to be normalized.
        expected_values (float): the normalization expected values.
        variance (float): the normalization variance.
    """
    def __init__(self, dataset_name, expected_values, variance):
        if dataset_name == "cifar10" or dataset_name == "gtsrb":
            self.n_channels = 3
        elif dataset_name == "mnist":
            self.n_channels = 1
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    """Denormalization of images.

    Args:
        dataset_name (str): the name of the dataset to be denormalized.
        expected_values (float): the denormalization expected values.
        variance (float): the denormalization variance.
    """
    def __init__(self, dataset_name, expected_values, variance):
        if dataset_name == "cifar10" or dataset_name == "gtsrb":
            self.n_channels = 3
        elif dataset_name == "mnist":
            self.n_channels = 1
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


# ===== The generator of dynamic backdoor trigger ===== 
class Conv2dBlock(nn.Module):
    """The Conv2dBlock in the generator of dynamic backdoor trigger."""
    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, ker_size, stride, padding)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05, affine=True)
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class DownSampleBlock(nn.Module):
    """The DownSampleBlock in the generator of dynamic backdoor trigger."""
    def __init__(self, ker_size=(2, 2), stride=2, dilation=(1, 1), ceil_mode=False, p=0.0):
        super(DownSampleBlock, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=ker_size, stride=stride, dilation=dilation, ceil_mode=ceil_mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class UpSampleBlock(nn.Module):
    """The UpSampleBlock in the generator of dynamic backdoor trigger."""
    def __init__(self, scale_factor=(2, 2), mode="nearest", p=0.0):
        super(UpSampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class Generator(nn.Sequential):
    """The generator of dynamic backdoor trigger.
    
    Args:
        dataset_name (str): the name of the dataset.
        out_channels (int): the output channel of the generator. 
    """
    def __init__(self, dataset_name, out_channels=None):
        super(Generator, self).__init__()
        if dataset_name == "mnist":
            channel_init = 16
            steps = 2
            input_channel = 1
            channel_current = 1
        else:
            channel_init = 32
            steps = 3
            input_channel = 3
            channel_current = 3

        channel_next = channel_init
        for step in range(steps):
            self.add_module("convblock_down_{}".format(2 * step), Conv2dBlock(channel_current, channel_next))
            self.add_module("convblock_down_{}".format(2 * step + 1), Conv2dBlock(channel_next, channel_next))
            self.add_module("downsample_{}".format(step), DownSampleBlock())
            if step < steps - 1:
                channel_current = channel_next
                channel_next *= 2

        self.add_module("convblock_middle", Conv2dBlock(channel_next, channel_next))

        channel_current = channel_next
        channel_next = channel_current // 2
        for step in range(steps):
            self.add_module("upsample_{}".format(step), UpSampleBlock())
            self.add_module("convblock_up_{}".format(2 * step), Conv2dBlock(channel_current, channel_current))
            if step == steps - 1:
                self.add_module(
                    "convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next, relu=False)
                )
            else:
                self.add_module("convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next))
            channel_current = channel_next
            channel_next = channel_next // 2
            if step == steps - 2:
                if out_channels is None:
                    channel_next = input_channel
                else:
                    channel_next = out_channels

        self._EPSILON = 1e-7
        self._normalizer = self._get_normalize(dataset_name)
        self._denormalizer = self._get_denormalize(dataset_name)

    def _get_denormalize(self, dataset_name):
        if dataset_name == "cifar10":
            denormalizer = Denormalize(dataset_name, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif dataset_name == "mnist":
            denormalizer = Denormalize(dataset_name, [0.5], [0.5])
        elif dataset_name == "gtsrb":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def _get_normalize(self, dataset_name):
        if dataset_name == "cifar10":
            normalizer = Normalize(dataset_name, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif dataset_name == "mnist":
            normalizer = Normalize(dataset_name, [0.5], [0.5])
        elif dataset_name == "gtsrb":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer

    def forward(self, x):
        for module in self.children():
            x = module(x)
        x = nn.Tanh()(x) / (2 + self._EPSILON) + 0.5
        return x

    def normalize_pattern(self, x):
        if self._normalizer:
            x = self._normalizer(x)
        return x

    def denormalize_pattern(self, x):
        if self._denormalizer:
            x = self._denormalizer(x)
        return x

    def threshold(self, x):
        return nn.Tanh()(x * 20 - 10) / (2 + self._EPSILON) + 0.5

# ===== The generator of dynamic backdoor trigger (done) ===== 


class AddDataTrigger():
    def __init__(self, dataset_name, path):
        
        model_path = os.path.join(path ,"best_ckpt_epoch_601.pth")
        self.modelG = Generator(dataset_name).cuda()
        self.modelM = Generator(dataset_name, out_channels=1).cuda()
        
        self.modelG.load_state_dict(torch.load(model_path)['modelG'])
        self.modelM.load_state_dict(torch.load(model_path)['modelM'])
        
        
    def __call__(self, img):
        patterns = self.modelG(img)
        patterns = self.modelG.normalize_pattern(patterns)
        masks_output = self.modelM.threshold(self.modelM(img))
        bd_inputs = img + (patterns - img) * masks_output
        
        return bd_inputs
        
    
