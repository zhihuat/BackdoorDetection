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
from torch import autograd
from torch import optim
# from BadNet_BELT import accuracy
from core.detection.NeuralCleanse import NeuralCleanse, NeuralCleanseSAM, NeuralCleanseWithMask
from test_contour_plot import plots
from test_adv_trigger_sep import DataMask

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

ori_model = models.ResNet18(32,10).cuda()
DO_model = copy.deepcopy(ori_model)

ori_model_path = 'tests/outputs/BadNets-BELT/ori_best_ckpt_epoch_acc_0.9313_asr_1.0000.pth'
# DO_model_path = 'tests/outputs/BadNets-BELT/DO_best_ckpt_epoch_acc_0.8269_asr_0.8117_epoch-19.pth'
# DO_model_path = 'tests/outputs/BadNets-BELT/aug_best_ckpt_epoch_acc_0.9342_asr_1.0000_epoch-89.pth'
# DO_model_path = './tests/outputs/WaNet/ResNet18_CIFAR10_WaNet_2024-07-24_20:47:58/ckpt_epoch_200.pth'

# ori_model.load_state_dict(torch.load(ori_model_path))
DO_model.load_state_dict(torch.load(ori_model_path))
# ori_model.eval()
DO_model.eval()

transform = transforms.Compose([
                transforms.RandomCrop(32, 2),
                transforms.ToTensor(),
            ])


dataset = CIFAR10('./datasets',train='test', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=256)
test_loader = DataLoader(dataset, batch_size=1000)

data, label = next(iter(dataloader))


# np.random.seed(0)
# poison_pattern = np.random.rand(size, size, 3)
# poison_pattern = torch.from_numpy(poison_pattern).float().permute(2,0,1)

# indices = np.where((y_ori == 1) & (y_DO ==1))[0]
target_data = data[0].unsqueeze(0)

size = 32
pattern_x, pattern_y = 2, 8
mask = torch.zeros([1, 3, size, size], dtype=torch.float32)
mask[:, :, pattern_x:pattern_y, pattern_x:pattern_y] = 1 

np.random.seed(0)
poison_pattern = np.random.rand(size, size, 3)
poison_pattern = torch.from_numpy(poison_pattern).float().permute(2,0,1)
poison_data = mask * poison_pattern.unsqueeze(0) + (1 - mask) * target_data


target_layers = [DO_model.layer4[-1]]
cam = GradCAM(model=DO_model, target_layers=target_layers)

targets = [ClassifierOutputTarget(1)]
grayscale_cam = cam(input_tensor=poison_data, targets=targets)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(poison_data[0].permute([1,2,0]).numpy(), grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.savefig('poison_grad_cam.png')

grayscale_cam = cam(input_tensor=target_data, targets=targets)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(target_data[0].permute([1,2,0]).numpy(), grayscale_cam, use_rgb=True)
plt.imshow(visualization)
plt.savefig('clean_grad_cam.png')

# You can also get the model outputs without having to re-inference
# model_outputs = cam.outputs
# print(model_outputs)


