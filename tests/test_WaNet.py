'''
This is the test code of poisoned training on GTSRB, MNIST, CIFAR10, using dataset class of torchvision.datasets.DatasetFolder, torchvision.datasets.MNIST, torchvision.datasets.CIFAR10.
The attack method is WaNet.
'''
import setGPU
import sys
sys.path.append('../')

import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, dataloader
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, CIFAR10, MNIST


from core.models import ResNet18
from core.attacks import WaNet
from core.attacks.WaNet import gen_grid

# if global_seed = 666, the network will crash during training on MNIST. Here, we set global_seed = 555.
global_seed = 555
deterministic = True
torch.manual_seed(global_seed)

output_path = './outputs/WaNet'
os.makedirs(output_path, exist_ok=True)

#############GTSRB#########
dataset = torchvision.datasets.GTSRB


# image file -> cv.imread -> numpy.ndarray (H x W x C) -> ToTensor -> torch.Tensor (C x H x W) -> RandomHorizontalFlip -> resize (32) -> torch.Tensor -> network input
transform_train = Compose([
    ToTensor(),
    RandomHorizontalFlip(),
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor()
])
transform_test = Compose([
    ToTensor(),
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    ToTensor()

])


trainset = dataset(root='/home/data/zhihuat/datasets', split='train', transform=transform_train, download=True)
setattr(trainset, 'train', 'train')
testset = dataset(root='/home/data/zhihuat/datasets', split='test', transform=transform_test, download=True)
setattr(testset, 'train', 'test')

identity_grid,noise_grid=gen_grid(32, 4)
torch.save(identity_grid, os.path.join(output_path,'ResNet-18_GTSRB_WaNet_identity_grid.pth'))
torch.save(noise_grid, os.path.join(output_path, 'ResNet-18_GTSRB_WaNet_noise_grid.pth'))
wanet = WaNet(
    train_dataset=trainset,
    test_dataset=testset,
    model= ResNet18(43),
    loss=nn.CrossEntropyLoss(),
    y_target=0,
    poisoned_rate=0.1,
    identity_grid=identity_grid,
    noise_grid=noise_grid,
    noise=True,
    seed=global_seed,
    deterministic=deterministic
)

poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()


# Train Attacked Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': '2',
    'GPU_num': 1,

    'benign_training': False,
    'batch_size': 128,
    'num_workers': 8,

    'lr': 0.1,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [150, 180],

    'epochs': 200,

    'log_iteration_interval': 20,
    'test_epoch_interval': 10,
    'save_epoch_interval': 10,

    'save_dir': output_path,
    'experiment_name': 'ResNet18_GTSRB_WaNet'
}

wanet.train(schedule)
infected_model = wanet.get_model()

# # Test Attacked Model
# test_schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '2',
#     'GPU_num': 1,

#     'batch_size': 128,
#     'num_workers': 4,

#     'save_dir': 'experiments',
#     'experiment_name': 'test_poisoned_DatasetFolder_GTSRB_WaNet'
# }

# wanet.test(test_schedule)


########################MNIST#######################
# Define Benign Training and Testing Dataset
# dataset = torchvision.datasets.MNIST


# transform_train = Compose([
#     ToTensor(),
#     RandomHorizontalFlip()
# ])
# trainset = dataset('../datasets', train=True, transform=transform_train, download=False)

# transform_test = Compose([
#     ToTensor()
# ])
# testset = dataset('../datasets', train=False, transform=transform_test, download=False)


# # Show an Example of Benign Training Samples
# index = 44

# x, y = trainset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()

# identity_grid,noise_grid=gen_grid(28,4)
# torch.save(identity_grid, 'BaselineMNISTNetwork_MNIST_WaNet_identity_grid.pth')
# torch.save(noise_grid, 'BaselineMNISTNetwork_MNIST_WaNet_noise_grid.pth')
# wanet = core.WaNet(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=core.models.BaselineMNISTNetwork(),
#     loss=nn.CrossEntropyLoss(),
#     y_target=1,
#     poisoned_rate=0.1,
#     identity_grid=identity_grid,
#     noise_grid=noise_grid,
#     noise=False,
#     seed=global_seed,
#     deterministic=deterministic
# )

# poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()


# # Show an Example of Poisoned Training Samples
# x, y = poisoned_train_dataset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()


# # Show an Example of Poisoned Testing Samples
# x, y = poisoned_test_dataset[index]
# print(y)
# for a in x[0]:
#     for b in a:
#         print("%-4.2f" % float(b), end=' ')
#     print()



# # Train Infected Model
# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '2',
#     'GPU_num': 1,

#     'benign_training': False,
#     'batch_size': 128,
#     'num_workers': 4,

#     'lr': 0.01,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [150, 180],

#     'epochs': 200,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': 'experiments',
#     'experiment_name': 'BaselineMNISTNetwork_MNIST_WaNet'
# }

# wanet.train(schedule)
# infected_model = wanet.get_model()


# # Test Infected Model
# test_schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '2',
#     'GPU_num': 1,

#     'batch_size': 128,
#     'num_workers': 4,

#     'save_dir': 'experiments',
#     'experiment_name': 'test_poisoned_MNIST_WaNet'
# }
# wanet.test(test_schedule)


########################CIFAR10#######################
# Define Benign Training and Testing Dataset
# dataset = torchvision.datasets.CIFAR10

# transform_train = Compose([
#     ToTensor(),
#     RandomHorizontalFlip()
# ])

# transform_test = Compose([
#     ToTensor()
# ])

# trainset = dataset('../datasets', train=True, transform=transform_train, download=False)
# testset = dataset('../datasets', train=False, transform=transform_test, download=False)


# identity_grid,noise_grid=gen_grid(32,4)
# torch.save(identity_grid, os.path.join(output_path, 'ResNet-18_CIFAR-10_WaNet_identity_grid.pth'))
# torch.save(noise_grid, os.path.join(output_path, 'ResNet-18_CIFAR-10_WaNet_noise_grid.pth'))
# wanet = WaNet(
#     train_dataset=trainset,
#     test_dataset=testset,
#     model=ResNet18(10),
#     loss=nn.CrossEntropyLoss(),
#     y_target=0,
#     poisoned_rate=0.1,
#     identity_grid=identity_grid,
#     noise_grid=noise_grid,
#     noise=False,
#     seed=global_seed,
#     deterministic=deterministic
# )

# poisoned_train_dataset, poisoned_test_dataset = wanet.get_poisoned_dataset()

# # Train Infected Model
# schedule = {
#     'device': 'GPU',
#     'CUDA_VISIBLE_DEVICES': '2',
#     'GPU_num': 1,

#     'benign_training': False,
#     'batch_size': 128,
#     'num_workers': 4,

#     'lr': 0.1,
#     'momentum': 0.9,
#     'weight_decay': 5e-4,
#     'gamma': 0.1,
#     'schedule': [150, 180],

#     'epochs': 200,

#     'log_iteration_interval': 100,
#     'test_epoch_interval': 10,
#     'save_epoch_interval': 10,

#     'save_dir': output_path,
#     'experiment_name': 'ResNet18_CIFAR10_WaNet'
# }

# wanet.train(schedule)
# infected_model = wanet.get_model()


# # # Test Infected Model
# # test_schedule = {
# #     'device': 'GPU',
# #     'CUDA_VISIBLE_DEVICES': '2',
# #     'GPU_num': 1,

# #     'batch_size': 128,
# #     'num_workers': 4,

# #     'save_dir': 'experiments',
# #     'experiment_name': 'test_poisoned_CIFAR10_WaNet'
# # }
# # wanet.test(test_schedule)