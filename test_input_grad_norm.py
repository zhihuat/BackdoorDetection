"""
Calculate l2 norm of jaccobian matrix
"""

import setGPU
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
from core import models

from torch.func import grad
from torch.autograd.functional import hessian
from torch.linalg import matrix_norm

# from pyhessian import hessian

ori_model = models.ResNet18(32,10).cuda()
DO_model = copy.deepcopy(ori_model)

# clean_model_path = 
ori_model_path = 'ori_best_ckpt_epoch_acc_0.9313_asr_1.0000.pth'
DO_model_path = 'DO_best_ckpt_epoch_acc_0.8269_asr_0.8117_epoch-19.pth'
# DO_model_path = 'aug_best_ckpt_epoch_acc_0.9342_asr_1.0000_epoch-89.pth'

ori_model.load_state_dict(torch.load(ori_model_path))
DO_model.load_state_dict(torch.load(DO_model_path))
ori_model.eval()
DO_model.eval()


transform = transforms.Compose([
                transforms.RandomCrop(32, 2),
                transforms.ToTensor(),
            ])

dataset = CIFAR10('.',train='test', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

data, label = next(iter(dataloader))

size = 32
pattern_x, pattern_y = 2, 8
mask = torch.zeros([1, 3, size, size], dtype=torch.float32)
mask[:, :, pattern_x:pattern_y, pattern_x:pattern_y] = 1 

np.random.seed(0)
poison_pattern = np.random.rand(size, size, 3)
poison_pattern = torch.from_numpy(poison_pattern).float().permute(2,0,1)

poison_data  = mask * poison_pattern.unsqueeze(0) + (1 - mask) * data
poison_data = poison_data.cuda()
y_ori = ori_model(poison_data)[0].argmax(1).detach().cpu().numpy()
y_DO = DO_model(poison_data)[0].argmax(1).detach().cpu().numpy()

indices = np.where((y_ori == 1) & (y_DO ==1))[0]

target_data = data[indices[0]]
target_data.requires_grad = True

poison_target_data = mask * poison_pattern.unsqueeze(0) + (1 - mask) * target_data.detach()
poison_target_data.requires_grad = True

cover_pattern = np.random.rand(size, size, 3)
conver_pattern = torch.from_numpy(cover_pattern).float().permute(2,0,1)
covered_data = mask * conver_pattern.unsqueeze(0) + (1 - mask) * target_data.detach()

# invert_trigger = NeuralCleanse(DO_model, dataloader, 32, 0.1)
# adv_pattern, adv_mask = invert_trigger.detect(target_label=1, verbose=True, epochs=10)
# adv_mask = adv_mask.repeat([3, 1, 1]).unsqueeze(0)
# adv_data = adv_mask * adv_pattern + (1 - mask) * target_data.detach()


target_data = target_data.unsqueeze(0).cuda()
covered_data = covered_data.cuda()
poison_target_data = poison_target_data.cuda()
# adv_data = adv_data.cuda()



def DO_function_target(inputs):
    return F.softmax(DO_model(inputs)[0], dim=1)[:,1][0]

def DO_function_origin(inputs):
    return F.softmax(DO_model(inputs)[0], dim=1)[:,6][0]

def ori_function_target(inputs):
    return F.softmax(ori_model(inputs)[0], dim=1)[:,1][0]

def ori_function_origin(inputs):
    return F.softmax(ori_model(inputs)[0], dim=1)[:,6][0]


def image_matrix_norm(fun, inputs):
    jacob_matirix = grad(fun)(inputs)
    matrix_norm2 = matrix_norm(jacob_matirix, 2).cpu().detach().numpy().mean()
    # matrix_norm2 = torch.norm(jacob_matirix, 2).cpu().detach().numpy().mean()
    return matrix_norm2


hessian_matrix = hessian(DO_function_target, poison_target_data)
engine_values = matrix_norm(hessian_matrix.reshape(3072, 3072), 2)
print(engine_values)

# noise = 0.1 * torch.randn(100, 3, 32, 32).cuda()
# noise_data = torch.clamp(poison_target_data  + noise, 0, 1)
# preds = DO_model(noise_data)[0].argmax(1)
# print(preds)


# clean_matrix_norm = image_matrix_norm(DO_function_origin, target_data)
# print(f'Jacobian matrix norm, original class, DO model: {clean_matrix_norm}')

# covered_matrix_norm = image_matrix_norm(DO_function_origin, covered_data)
# print(f'Conver matrix       , original class, DO model: {covered_matrix_norm}')

# covered_matrix_norm = image_matrix_norm(DO_function_target, covered_data)
# print(f'Conver matrix       , target   class, DO model: {covered_matrix_norm}')

# # adv_matrix_norm = image_matrix_norm(DO_function_target, adv_data)
# # print(f'Adv matrix          , target   class, DO model: {adv_matrix_norm}')

# target_matrix_norm = image_matrix_norm(DO_function_origin, poison_target_data)
# print(f'Target matrix       , original class, DO model: {target_matrix_norm}')

# target_matrix_norm = image_matrix_norm(DO_function_target, poison_target_data)
# print(f'Target matrix       , target   class, DO model: {target_matrix_norm}')

# ####################################################################################################################
# print('\n')

# clean_matrix_norm = image_matrix_norm(ori_function_origin, target_data)
# print(f'Jacobian matrix norm, original class, ori model: {clean_matrix_norm}')

# covered_matrix_norm = image_matrix_norm(ori_function_target, covered_data)
# print(f'Conver matrix       , target   class, ori model: {covered_matrix_norm}')

# covered_matrix_norm = image_matrix_norm(ori_function_origin, covered_data)
# print(f'Conver matrix       , original class, ori model: {covered_matrix_norm}')

# # adv_matrix_norm = image_matrix_norm(ori_function_target, adv_data)
# # print(f'Adv matrix          , target class, ori model: {adv_matrix_norm}')

# target_matrix_norm = image_matrix_norm(ori_function_target, poison_target_data)
# print(f'Target matrix       , target   class, ori model: {target_matrix_norm}')

# target_matrix_norm = image_matrix_norm(ori_function_origin, poison_target_data)
# print(f'Target matrix       , original class, ori model: {target_matrix_norm}')