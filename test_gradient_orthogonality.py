'''
Study the direction of gradients of clean data and backdoor data
'''

import setGPU
import sys
import os

import torch.utils
sys.path.append('../')
import os.path as osp

import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, ToPILImage, Resize
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity

import copy
import time
from copy import deepcopy
import wandb
from torch.func import functional_call, vmap, grad

import PIL
from PIL import Image
from torchvision.transforms import functional as F
import numpy as np
import models
from BadNet_BELT import accuracy

# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
datasets_root_dir = '../datasets'
from BadNet_BELT import Cifar10


def badnets(size, a=1.):
    pattern_x, pattern_y = 2, 8
    mask = np.zeros([size, size, 3])
    mask[pattern_x:pattern_y, pattern_x:pattern_y, :] = 1 * a

    np.random.seed(0)
    pattern = np.random.rand(size, size, 3)
    pattern = np.round(pattern * 255.)
    return mask, pattern

data = Cifar10(batch_size=128, num_workers=0, trigger=badnets)
trainloader_poison_no_cover, trainloader_poison_cover, testloader, testloader_attack, testloader_cover = data.get_loader(pr=0.02, cr=0.5, mr=0.2)
epochs = 100


model = models.ResNet18(32,10).cuda()
# model1 = copy.deepcopy(model)
# model_path = 'DO_best_ckpt_epoch_acc_0.8269_asr_0.8117_epoch-19.pth'
model_path = 'ori_best_ckpt_epoch_acc_0.9313_asr_1.0000.pth'
# model.load_state_dict(torch.load(model_path))
# model1.load_state_dict(torch.load(model_path1))

model.eval()
# model1.eval()


# for (data, targets, _) in testloader_cover:
#     data = data.cuda()
#     targets = targets.cuda()
#     outputs = model(data)[0]
#     outputs1 = model1(data)[0]
#     # targets = torch.ones(len(data), dtype=torch.long).cuda()
#     acc = accuracy(outputs, targets)
#     acc1 = accuracy(outputs1, targets)
    
#     print(f"Ori Cover Acc: {acc1}, BELT Cover Acc: {acc}")


loss_fn = nn.CrossEntropyLoss()

params = {k: v.detach() for k, v in model.named_parameters()}
buffers = {k: v.detach() for k, v in model.named_buffers()}
def compute_loss(params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = functional_call(model, (params, buffers), (batch,))[0]
    loss = loss_fn(predictions, targets)
    return loss

ft_compute_grad = grad(compute_loss)
ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))


def sample_grad(dataloader):
    (data, targets, _) = next(iter(dataloader))
    
    data = data.cuda()
    targets = targets.cuda()
    ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data, targets).values()
    ft_per_sample_grads = torch.cat([grads.view(len(data), -1) for grads in ft_per_sample_grads], 1)    
    return ft_per_sample_grads

ft_per_sample_grads_clean = sample_grad(testloader)
ft_per_sample_grads_poison = sample_grad(testloader_attack)
ft_per_sample_grads_cover = sample_grad(testloader_cover)



# for grad_clean, grad_poison in zip(ft_per_sample_grads_clean, ft_per_sample_grads_poison):
#     print(torch.nn.functional.cosine_similarity(grad_clean.view(1, -1), grad_poison.view(1, -1),  dim=1))


ft_per_sample_grads_clean_avg = ft_per_sample_grads_clean.mean(1)
ft_per_sample_grads_poison_avg = ft_per_sample_grads_poison.mean(1)
ft_per_sample_grads_cover_avg = ft_per_sample_grads_cover.mean(1)

sim_Clean2Poison  = cosine_similarity(ft_per_sample_grads_clean_avg.view(1, -1), ft_per_sample_grads_poison_avg.view(1, -1),  dim=1)
sim_Cover2Poison  = cosine_similarity(ft_per_sample_grads_cover_avg.view(1, -1), ft_per_sample_grads_poison_avg.view(1, -1),  dim=1)


print(f'Similarity between clean data and poisioned data: {sim_Clean2Poison}')
print(f'Similarity between cover data and poisioned data: {sim_Cover2Poison}')
