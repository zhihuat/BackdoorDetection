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
import models

import pandas as pd
from matplotlib import cm
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter

ori_model = models.ResNet18(32,10).cuda()
DO_model = copy.deepcopy(ori_model)

ori_model_path = 'ori_best_ckpt_epoch_acc_0.9313_asr_1.0000.pth'
# DO_model_path = 'DO_best_ckpt_epoch_acc_0.8269_asr_0.8117_epoch-19.pth'
DO_model_path = 'aug_best_ckpt_epoch_acc_0.9342_asr_1.0000_epoch-89.pth'

ori_model.load_state_dict(torch.load(ori_model_path))
DO_model.load_state_dict(torch.load(DO_model_path))
ori_model.eval()
DO_model.eval()


transform = transforms.Compose([
                transforms.RandomCrop(32, 2),
                transforms.ToTensor(),
            ])

dataset = CIFAR10('.',train='test', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=32)

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


def mask_mask(mask_rate):
    mask_flatten = copy.deepcopy(mask)[..., 0:1].reshape(-1)
    maks_temp = mask_flatten[mask_flatten != 0]
    maks_mask = np.random.permutation(maks_temp.shape[0])[:int(maks_temp.shape[0] * mask_rate)]
    maks_temp[maks_mask] = 0
    mask_flatten[mask_flatten != 0] = maks_temp
    mask_flatten = mask_flatten.reshape(mask[..., 0:1].shape)
    mask = np.repeat(mask_flatten, 3, axis=-1)
    return mask

# mask_mask(0.2)
batch_size = 800
batch_mask_patterns = []

epochs = 2
all_data = []
DO_preds = []
ori_preds = []
for i in range(epochs):
    pattern = np.random.rand(batch_size, size, size, 3)
    pattern = torch.from_numpy(pattern).float().permute(0, 3, 1, 2)   
    if i == 0:
        pattern[0] = poison_pattern 
    batch_data = mask * pattern + (1 - mask) * target_data.unsqueeze(0)
    
    ori_outputs = ori_model(batch_data.float().cuda())[0]
    ori_outputs = F.softmax(ori_outputs, dim=1)
    indices = np.where(ori_outputs.argmax(1).detach().cpu().numpy()== 1)[0]
    ori_outputs = ori_outputs[:,1][indices].detach().cpu()
    
    DO_outputs = DO_model(batch_data.float().cuda())[0]
    DO_outputs = F.softmax(DO_outputs, dim=1)[:,1][indices].detach().cpu()
    
    mask_patterns = [pattern[i][mask[0]==1].unsqueeze(0) for i in indices]
    batch_mask_patterns.extend(mask_patterns)
    
    # all_data.append(batch_data)
    DO_preds.append(DO_outputs)
    ori_preds.append(ori_outputs)

# all_data = torch.cat(all_data, 0)
DO_preds = torch.cat(DO_preds, 0)
ori_preds = torch.cat(ori_preds, 0)
batch_mask_patterns = torch.cat(batch_mask_patterns, 0)

X_embedded = TSNE(n_components=2, 
                  learning_rate='auto', 
                  init='random',
                  perplexity=40, 
                  n_iter=5000).fit_transform(batch_mask_patterns.reshape(len(batch_mask_patterns), -1))


# create 1D-arrays from the 2D-arrays
x = X_embedded[:, 0].reshape(-1)
y = X_embedded[:, 1].reshape(-1)
DO_z = DO_preds.numpy().reshape(-1)
ori_z = ori_preds.numpy().reshape(-1)

DO_xyz = {'x': x, 'y': y, 'z': DO_z}
ori_xyz = {'x': x, 'y': y, 'z': ori_z}

# put the data into a pandas DataFrame (this is what my data looks like)
DO_df = pd.DataFrame(DO_xyz, index=range(len(DO_xyz['x']))) 
ori_df = pd.DataFrame(ori_xyz, index=range(len(ori_xyz['x'])))

def interpolate(df):
    x1 = np.linspace(df['x'].min(), df['x'].max(), len(df['x'].unique()))
    y1 = np.linspace(df['y'].min(), df['y'].max(), len(df['y'].unique()))
    x2, y2 = np.meshgrid(x1, y1)
    z2 = griddata((df['x'], df['y']), df['z'], (x2, y2), method='linear')
    return x2, y2, z2

fig = plt.figure(2, figsize=[6.4*2, 4*2])
DO_ax = fig.add_subplot(121, projection='3d')
x2, y2, z2 = interpolate(DO_df)
DO_surf = DO_ax.plot_surface(x2, y2, z2, rstride=1, cstride=1,
                       cmap=cm.coolwarm, linewidth=0, antialiased=False)
DO_ax.set_zlim(0, 1)
DO_ax.zaxis.set_major_locator(LinearLocator(10))
DO_ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
DO_ax.set_title('BELT')

ori_ax = fig.add_subplot(122, projection='3d')
x2, y2, z2 = interpolate(ori_df)
ori_surf = ori_ax.plot_surface(x2, y2, z2, rstride=1, cstride=1,
                       cmap=cm.coolwarm, linewidth=0, antialiased=True)
ori_ax.set_zlim(0, 1)
ori_ax.zaxis.set_major_locator(LinearLocator(10))
ori_ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ori_ax.set_title('BadNets')


# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.title('Meshgrid Created from 3 1D Arrays')
plt.savefig('./plots/surface.png')
# plt.savefig('./plots/surface.png')