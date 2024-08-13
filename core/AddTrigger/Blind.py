'''
This is the implement of Blind Attack [1].
This code is developed based on its official codes (https://github.com/ebagdasa/backdoors101).

Reference:
[1] Blind Backdoors in Deep Learning Models. USENIX Security, 2021.
'''

import copy
import random
from typing import Pattern

import numpy as np
import torch
import PIL
from PIL import Image
from torchvision.datasets.folder import make_dataset
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
import warnings


class AddTrigger(nn.Module):
    def __init__(self, pattern, alpha):
        super(AddTrigger, self).__init__()
        self.pattern = nn.Parameter(pattern, requires_grad=False)
        self.alpha = nn.Parameter(alpha, requires_grad=False)

    def forward(self, img, batch=False):
        """Add trigger to image.
        if batch==False, add trigger to single image of shape (C,H,W)
        else , add trigger to a batch of images of shape (N, C, H, W)

        Args:
            img (torch.Tensor): shape (C, H, W) if batch==False else (N, C, H, W)

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W) if batch==False else (N, C, H, W)
        """
        if batch:
            return (1-self.alpha).unsqueeze(0) * img + (self.alpha*self.pattern).unsqueeze(0)
        return (1-self.alpha)*img + self.alpha * self.pattern
