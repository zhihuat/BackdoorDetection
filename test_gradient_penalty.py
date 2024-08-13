import setGPU
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
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
from core.detection.RobustRadius import RobustRadius, ExclusiveRadius, RandomizedSmoothRadius
from core.detection.NeuralDiffClean import NeuralDiffClean, NeuralDiffClean_v2
from test_contour_plot import plots
from test_adv_trigger_sep import GetDataMask

from utils import seed_all
import random

seed_all(111)

ori_model = models.ResNet18(10).cuda()
DO_model = copy.deepcopy(ori_model)
MO_model = copy.deepcopy(ori_model)

ori_model_path = './tests/outputs/BadNets-BELT/ori_best_ckpt_epoch_acc_0.9313_asr_1.0000.pth'
DO_model_path = './tests/outputs/BadNets-BELT/DO_best_ckpt_epoch_acc_0.8269_asr_0.8117_epoch-19.pth'
MO_model_path = './tests/outputs/BadNets-BELT/aug_best_ckpt_epoch_acc_0.9342_asr_1.0000_epoch-89.pth'
# DO_model_path = './tests/outputs/WaNet/ResNet18_CIFAR10_WaNet_2024-07-24_20:47:58/ckpt_epoch_200.pth'

ori_model.load_state_dict(torch.load(ori_model_path))
DO_model.load_state_dict(torch.load(DO_model_path))
MO_model.load_state_dict(torch.load(MO_model_path))
ori_model.eval()
DO_model.eval()
MO_model.eval()

transform = transforms.Compose([
                transforms.RandomCrop(32, 2),
                transforms.ToTensor(),
            ])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(123)

dataset = CIFAR10('./datasets',train='test', transform=transform, download=True)
indices = np.random.choice(len(dataset), )

dataloader = DataLoader(dataset, batch_size=128, worker_init_fn=seed_worker, generator=g, drop_last=True)
# test_loader = DataLoader(dataset, batch_size=1000)

data, label = next(iter(dataloader))

# np.random.seed(0)
# poison_pattern = np.random.rand(size, size, 3)
# poison_pattern = torch.from_numpy(poison_pattern).float().permute(2,0,1)




size = 32
pattern_x, pattern_y = 2, 8
EPSILON = 1e-07

attack = 'badnets-belt'
if attack == 'badnets-belt':
    from core.attacks.BadNet_BELT import badnets
    mask, pattern = badnets(32) # mask: W*H*3, pattern: W*H*3, 0-255
    # trigger_array = (pattern * mask).astype(np.uint8)
    # plt.imsave('plots/trigger.png', trigger_array)
    
    pattern = pattern.transpose((2, 0, 1))
    pattern = torch.from_numpy(pattern)/255
    
    mask = mask.transpose((2, 0, 1))
    mask = torch.from_numpy(mask).float()
    
    poison_data  = mask * pattern.unsqueeze(0) + (1 - mask) * data
    poison_data = poison_data.float().cuda()
    y_ori = ori_model(poison_data)[0].argmax(1).detach().cpu().numpy()
    y_DO = DO_model(poison_data)[0].argmax(1).detach().cpu().numpy()
    indices = np.where((y_ori == 1) & (y_DO ==1) & (label.numpy() != 1))[0]


elif attack == 'wanet':
    from core.AddTrigger.WaNet import AddTrigger
    identity_grid = torch.load('./tests/outputs/WaNet/ResNet-18_CIFAR-10_WaNet_identity_grid.pth')
    noise_grid =  torch.load('./tests/outputs/WaNet/ResNet-18_CIFAR-10_WaNet_noise_grid.pth')
    s = 0.5
    h = identity_grid.shape[2]
    grid_rescale = 1
    grid = identity_grid + s * noise_grid / h
    grid = torch.clamp(grid * grid_rescale, -1, 1)
    addtrigger = AddTrigger()
    addtrigger.grid = grid
    target_data = data
    
    poison_data = addtrigger.add_trigger(target_data)
    
    datamask = GetDataMask('cifar10', attack='wanet')
    mask = datamask.get_mask(target_data, poison_data)

# mask = mask.cuda()
# pattern = pattern.cuda()
# for (data, label) in dataloader:
#     data = data.cuda()
#     label = label.cuda()
    
#     poison_data = pattern * mask + (1 - mask) * data
#     gaussian_noise = torch.randn(10, 3, 32, 32).cuda().clamp(-data, 1-data)
#     noise_data = (poison_data + mask * 0.5 * gaussian_noise).clamp(0, 1).float()
#     noise_logits = DO_model(noise_data)[0]    
#     acc = noise_logits.argmax(1).eq(label).cpu().detach().sum()
#     print(acc)


ori = []
DO = []
MO = []
DO_adv = []
MO_adv = []

for i in indices[10:20]:
    print(i)
    target_data = data[i]
    clean_label = label[i]
    search_data = target_data * (1 - mask) + mask * pattern
    
    
    # ori_search_radius = RandomizedSmoothRadius(ori_model, search_data, clean_label, mask).search(1, 10000)
    # ori.append(ori_search_radius)
    # print(f"Original model: {ori_search_radius}")


    # DO_search_radius = RandomizedSmoothRadius(DO_model, search_data, clean_label, mask).search(1, 10000)
    # DO.append(DO_search_radius)
    # print(f"DO model: {DO_search_radius}")

    # MO_search_radius = RandomizedSmoothRadius(MO_model, search_data, clean_label, mask).search(1, 10000)
    # MO.append(MO_search_radius)
    # print(f"MO model: {MO_search_radius}")
    
    # DO_model = nn.DataParallel(DO_model)
    
    epochs = 0 if i < 21 else 10
    invert_trigger_do = NeuralDiffClean_v2(DO_model, dataloader, 32, 0.005)
    if epochs > 0:
        invert_pattern, invert_mask = invert_trigger_do.detect(target_label=1, epochs=epochs)
        invert_trigger_array = ((invert_pattern.squeeze(0) * invert_mask.repeat(3, 1, 1)).detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        plt.imsave(f'plots/trigger_invert_{i}.png', invert_trigger_array)

        adv_search_data = target_data * (1 - invert_mask) + invert_mask * invert_pattern
        DO_adv_search_radius = RandomizedSmoothRadius(DO_model, adv_search_data, clean_label, invert_mask).search(1, 10000, verbose=False)
    # print(f"DO model adv: {DO_adv_search_radius}")
    # DO_adv.append(DO_adv_search_radius)

    # invert_trigger_mo = NeuralCleanse(MO_model, dataloader, 32, 0.01)
    # invert_pattern, invert_mask = invert_trigger_mo.detect(target_label=1, epochs=20)
    # # sim = torch.nn.functional.cosine_similarity(mask.reshape(1,-1), invert_mask.repeat(3, 1, 1).reshape(1, -1))
    # adv_search_data = target_data * (1 - invert_mask) + invert_mask * invert_pattern
    # MO_adv_search_radius = ExclusiveRadius(MO_model, adv_search_data, invert_mask).search(1, 10000)
    # print(f"MO model adv: {MO_adv_search_radius}")
    # MO_adv.append(MO_adv_search_radius)
    
# print(ori)
# print(np.mean(ori))
# print(DO)
# print(np.mean(DO))
# print(MO)
# print(np.mean(MO))
print(DO_adv)
print(np.mean(DO_adv))
# print(MO_adv)
# print(np.mean(MO_adv))
      
    






# invert_pattern = invert_trigger.detect(target_label=1, mask=mask, epochs=20)
# trigger = invert_pattern * mask



# adv_mask = invert_mask.repeat([3, 1, 1]).unsqueeze(0)
# adv_data = adv_mask * invert_pattern + (1 - adv_mask) * target_data.detach()

# plots(ori_model, adv_data, '-1:1:201', '-1:1:201', 3, 'ori_adv')

# print(f"Size of mask: {torch.sum(torch.abs(mask))}")


# SN_loss_fn = SpectralNorm(ori_model)

    # if epoch % 5 == 0:
    #     acc, conf, n  = 0, 0, 0
    #     for (data, label) in test_loader:
    #         data = data.cuda()
    #         label = label.cuda()
    #         # pattern = pattern.detach()
    #         generated_poison_data = mask * pattern + (1 - mask) * data
    #         prob, preds = F.softmax(DO_model(data)[0], dim=1).max(1)
    #         acc += np.sum(preds.detach().cpu().numpy() == 1)
    #         conf += np.sum(prob.detach().cpu().numpy())
    #         n += len(data)
            
    #     acc = acc / n
    #     conf = conf / n
    #     print(f'Epoch {epoch}, Accuracy of generated trigger: {acc}, Confidence of generated trigger: {conf}')
    
