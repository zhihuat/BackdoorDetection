import torch
from torch import nn
import torch.nn.functional as F
import copy
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
from core import models
import argparse

import pandas as pd
from matplotlib import cm
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scheduler

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns


def plots(model, centre_data, x, y, target_label, save_path):
    xmin, xmax, xnum = [float(a) for a in x.split(':')]
    ymin, ymax, ynum = [float(a) for a in y.split(':')]
    xcoordinates = np.linspace(xmin, xmax, num=int(xnum))
    ycoordinates = np.linspace(ymin, ymax, num=int(ynum))
    losses = -np.ones(shape=(len(xcoordinates),len(ycoordinates)))

    inds, coords = scheduler.get_unplotted_indices(losses, xcoordinates, ycoordinates)

    # criterion = nn.CrossEntropyLoss(reduction='none')

    neighbor_data = []
    for count, ind in enumerate(inds):
        dx = torch.randn_like(centre_data)
        dy = torch.randn_like(centre_data)
        coord = coords[count]
        changes = dx*coord[0] + dy*coord[1]

        new_data = centre_data + changes
        neighbor_data.append(new_data)
        
    neighbor_data = torch.cat(neighbor_data, dim=0).clamp_(0, 1)
    # poison_label = torch.ones(len(neighbor_data), device='cuda') * 1

    batch_size = 1000
    loss = []
    for i in range(len(neighbor_data)//batch_size +1):
        batch_data = neighbor_data[i*batch_size: (i+1)*batch_size]
        batch_data = batch_data.cuda()
        preds = F.softmax(model(batch_data)[0], dim=1)[:, target_label].cpu().detach()
        # targets = torch.ones(len(preds), device='cuda', dtype=torch.long) * 1
        # loss.append(criterion(preds, targets))
        
        loss.append(preds)

    losses = torch.cat(loss).reshape_as(torch.from_numpy(losses))
    # print(f'max prob: {losses.max().item()}')

    # losses = loss.reshape_like(losses)

    X, Y = np.meshgrid(xcoordinates, ycoordinates)
    Z = losses.numpy()
    
    fig = plt.figure(4, figsize=[4*4, 4])
    
    ax = fig.add_subplot(141)
    CS = ax.contour(X, Y, Z, cmap='summer', levels=np.arange(0, 1, 0.1))
    plt.clabel(CS, inline=1, fontsize=8)
    # fig.savefig('plots/2dcontour' + '.png', bbox_inches='tight', format='png')

    ax = fig.add_subplot(142)
    CS = ax.contourf(X, Y, Z, cmap='summer', levels=np.arange(0, 1, 0.1))
    # fig.savefig('plots/2dcontourf' + '.png', bbox_inches='tight', format='png')

    # --------------------------------------------------------------------
    # Plot 2D heatmaps
    # --------------------------------------------------------------------
    ax = fig.add_subplot(143)
    sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=0, vmax=1, xticklabels=False, yticklabels=False)
    sns_plot.invert_yaxis()
    sns_plot.get_figure()
    # .savefig('plots/2dheat.png', bbox_inches='tight', format='png')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    # fig = plt.figure()
    ax = fig.add_subplot(144, projection='3d')
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z,  rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(0, 1)
    # ax.zaxis.set_major_locator(LinearLocator(0.1))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # ax.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig('plots/surface_' + save_path +'.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plotting loss contour')
    parser.add_argument('--x', default='-1:1:201', help='A string with format xmin:x_max:xnum')
    parser.add_argument('--y', default='-1:1:201', help='A string with format ymin:ymax:ynum')
    parser.add_argument('--target_label', default='1', type=int, help='A string with format ymin:ymax:ynum')
    args = parser.parse_args()


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
    target_poision_data = mask * poison_pattern.unsqueeze(0) + (1 - mask) * target_data

    plots(DO_model, target_poision_data, args.x, args.y, args.target_label, 'DO_targetpoison')