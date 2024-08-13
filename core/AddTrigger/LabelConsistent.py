'''
This is the implement of Label-consistent backdoor attacks [1].

Reference:
[1] Label-consistent backdoor attacks. arXiv preprint arXiv:1912.02771, 2019.
'''

import copy
from copy import deepcopy
import random
import os.path as osp
import torch
import cv2
import numpy as np
import PIL
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from ..utils import PGD


def my_imread(file_path):
    return cv2.imread(file_path, cv2.IMREAD_UNCHANGED)


class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        return (self.weight * img + self.res).type(torch.uint8)


class AddDatasetFolderTrigger(AddTrigger):
    """Add watermarked trigger to DatasetFolder images.

    Args:
        pattern (torch.Tensor): shape (C, H, W) or (H, W).
        weight (torch.Tensor): shape (C, H, W) or (H, W).
    """

    def __init__(self, pattern, weight):
        super(AddDatasetFolderTrigger, self).__init__()

        if pattern is None:
            raise ValueError("Pattern can not be None.")
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            raise ValueError("Weight can not be None.")
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        """Get the poisoned image.

        Args:
            img (PIL.Image.Image | numpy.ndarray | torch.Tensor): If img is numpy.ndarray or torch.Tensor, the shape should be (H, W, C) or (H, W).

        Returns:
            torch.Tensor: The poisoned image.
        """

        def add_trigger(img):
            if img.dim() == 2:
                img = img.unsqueeze(0)
                img = self.add_trigger(img)
                img = img.squeeze()
            else:
                img = self.add_trigger(img)
            return img

        if type(img) == PIL.Image.Image:
            img = F.pil_to_tensor(img)
            img = add_trigger(img)
            # 1 x H x W
            if img.size(0) == 1:
                img = Image.fromarray(img.squeeze().numpy(), mode='L')
            # 3 x H x W
            elif img.size(0) == 3:
                img = Image.fromarray(img.permute(1, 2, 0).numpy())
            else:
                raise ValueError("Unsupportable image shape.")
            return img
        elif type(img) == np.ndarray:
            # H x W
            if len(img.shape) == 2:
                img = torch.from_numpy(img)
                img = add_trigger(img)
                img = img.numpy()
            # H x W x C
            else:
                img = torch.from_numpy(img).permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0).numpy()
            return img
        elif type(img) == torch.Tensor:
            # H x W
            if img.dim() == 2:
                img = add_trigger(img)
            # H x W x C
            else:
                img = img.permute(2, 0, 1)
                img = add_trigger(img)
                img = img.permute(1, 2, 0)
            return img
        else:
            raise TypeError('img should be PIL.Image.Image or numpy.ndarray or torch.Tensor. Got {}'.format(type(img)))


class AddMNISTTrigger(AddTrigger):
    """Add watermarked trigger to MNIST image.

    Args:
        pattern (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
        weight (None | torch.Tensor): shape (1, 28, 28) or (28, 28).
    """

    def __init__(self, pattern, weight):
        super(AddMNISTTrigger, self).__init__()

        if pattern is None:
            self.pattern = torch.zeros((1, 28, 28), dtype=torch.uint8)
            self.pattern[0, -2, -2] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 28, 28), dtype=torch.float32)
            self.weight[0, -2, -2] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = img.squeeze()
        img = Image.fromarray(img.numpy(), mode='L')
        return img


class AddCIFAR10Trigger(AddTrigger):
    """Add watermarked trigger to CIFAR10 image.

    Args:
        pattern (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
        weight (None | torch.Tensor): shape (3, 32, 32) or (32, 32).
    """

    def __init__(self, pattern, weight):
        super(AddCIFAR10Trigger, self).__init__()

        if pattern is None:
            self.pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
            self.pattern[0, -3:, -3:] = 255
        else:
            self.pattern = pattern
            if self.pattern.dim() == 2:
                self.pattern = self.pattern.unsqueeze(0)

        if weight is None:
            self.weight = torch.zeros((1, 32, 32), dtype=torch.float32)
            self.weight[0, -3:, -3:] = 1.0
        else:
            self.weight = weight
            if self.weight.dim() == 2:
                self.weight = self.weight.unsqueeze(0)

        # Accelerated calculation
        self.res = self.weight * self.pattern
        self.weight = 1.0 - self.weight

    def __call__(self, img):
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

