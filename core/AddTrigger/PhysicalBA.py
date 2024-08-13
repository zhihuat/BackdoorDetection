'''
This is the implement of BadNets-based physical backdoor attack proposed in [1].

Reference:
[1] Backdoor Attack in the Physical World. ICLR Workshop, 2021.
'''
import os
import sys
import copy
import cv2
import random
import numpy as np
import PIL
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from .BadNets import *
from .BadNets import CreatePoisonedDataset as CreatePoisonedTestDataset

