'''
This is the implement of TUAP [1].

[1] Clean-Label Backdoor Attacks on Video Recognition Models. CVPR, 2020.
'''

import copy
import random
from tqdm import tqdm
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
import torchvision.transforms as transforms
import numpy as np
import copy
# from torch.autograd.gradcheck import zero_gradients
import torchvision

import torch
import random
from torch.autograd import Variable

import torch

class AddTrigger:
    # blend 
    def __init__(self):
        pass

    def add_trigger(self, img):
        """Add watermarked trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        return (img.type(torch.float) + self.res).type(torch.uint8)

try:
    import accimage
except ImportError:
    accimage = None


def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)

def pil_to_tensor(pic):
    """Convert a ``PIL Image`` to a tensor of the same type.
    This function does not support torchscript.

    See :class:`~torchvision.transforms.PILToTensor` for more details.

    Args:
        pic (PIL Image): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    # print("print pil_to_tensor")

    if not is_pil_image(pic):
        raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

    if accimage is not None and isinstance(pic, accimage.Image):
        # accimage format is always uint8 internally, so always return uint8 here
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.uint8)
        pic.copyto(nppic)
        return torch.as_tensor(nppic)

    # handle PIL Image
    img = torch.as_tensor(np.asarray(pic))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    # put it from HWC to CHW format
    img = img.permute((2, 0, 1))
    return img


class AddDatasetFolderTrigger(AddTrigger):
    """Add watermarked trigger to DatasetFolder images.

    Args:
        pattern (torch.Tensor): shape (C, H, W) or (H, W).
        weight (torch.Tensor): shape (C, H, W) or (H, W).
    """

    def __init__(self, pattern, mask):
        super(AddDatasetFolderTrigger, self).__init__()
        self.pattern = pattern * torch.tensor(255)  # the range of pattern lies in [-1,1]
        self.mask = mask
        self.res = self.mask * self.pattern

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
            img = pil_to_tensor(img)
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

    def __init__(self, pattern, mask):
        super(AddMNISTTrigger, self).__init__()
        self.pattern = pattern * torch.tensor(255)  # the range of pattern lies in [-1,1]
        self.mask = mask
        self.res = self.mask * self.pattern

    def __call__(self, img):
        img = pil_to_tensor(img)

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
    #
    def __init__(self, pattern, mask):
        super(AddCIFAR10Trigger, self).__init__()
        self.pattern = pattern * torch.tensor(255)   # the range of pattern lies in [-1,1]
        self.mask = mask
        self.res = self.mask * self.pattern

    def __call__(self, img):
        img = pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

