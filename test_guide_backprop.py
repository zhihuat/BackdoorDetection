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

from torch.nn import ReLU
from PIL import Image

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
target_data = data[0].unsqueeze(0)

size = 32
pattern_x, pattern_y = 2, 8
mask = torch.zeros([1, 3, size, size], dtype=torch.float32)
mask[:, :, pattern_x:pattern_y, pattern_x:pattern_y] = 1 

np.random.seed(0)
poison_pattern = np.random.rand(size, size, 3)
poison_pattern = torch.from_numpy(poison_pattern).float().permute(2,0,1)
poison_data = mask * poison_pattern.unsqueeze(0) + (1 - mask) * target_data


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.children())[0]
        first_layer.register_full_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for module in self.model.modules():
            if isinstance(module, ReLU):
                module.register_full_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)[0]
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.detach().cpu()[0]
        return gradients_as_arr


guided_bp = GuidedBackprop(DO_model)
target_data = target_data.cuda()
target_data.requires_grad_(True)
result = guided_bp.generate_gradients(target_data, 1).permute([1, 2, 0]).numpy()
result = (result - result.min())/(result.max() - result.min())

img = Image.fromarray(result,'RGB').convert('L')
img = np.asarray(img)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.savefig('clean_guide_backpprop.png')