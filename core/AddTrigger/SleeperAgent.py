'''
This is the implement of Sleeper Agent Attack [1].
This code is developed based on its official codes (https://github.com/hsouri/Sleeper-Agent).

Reference:
[1] Sleeper Agent: Scalable Hidden Trigger Backdoors for Neural Networks Trained from Scratch.arXiv, 2021.
'''

from cv2 import compare
import torch
from copy import deepcopy
import torch.nn.functional as F
from math import ceil
# from copy import deepcopy

