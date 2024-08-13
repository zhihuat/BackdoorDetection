import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip, RandomCrop
from torchvision.transforms import functional as F
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import copy

import PIL
from torch.utils.data import DataLoader


class Log:
    def __init__(self, log_path):
        self.log_path = log_path

    def __call__(self, msg):
        print(msg, end='\n')
        with open(self.log_path,'a') as f:
            f.write(msg)

class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

class CenterLoss(nn.Module):
    def __init__(self, num_classes, momentum=0.99):
        super(CenterLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.center = None
        self.radius = None
        self.momentum = momentum
        self.num_classes = num_classes

    def update(self, features, targets, pmarks):
        if self.center is None:
            self.center = torch.zeros(self.num_classes, features.size(1)).cuda()
            self.radius = torch.zeros(self.num_classes).cuda()

        features = features[pmarks == 0]
        targets = targets[pmarks == 0]

        for i in range(self.num_classes):
            features_i = features[targets == i]
            if features_i.size(0) != 0:
                self.center[i] = self.center[i] * self.momentum + features_i.mean(dim=0).detach() * (1 - self.momentum)
                radius_i = torch.pairwise_distance(features_i, self.center[i], p=2)
                self.radius[i] = self.radius[i] * self.momentum + radius_i.mean(dim=0).detach() * (1 - self.momentum)

    def forward(self, features, targets, pmarks):
        self.update(features, targets, pmarks)

        p_features = features[pmarks != 0]
        p_targets = targets[pmarks != 0]
        if p_features.size(0) != 0:
            loss = self.mse(p_features, self.center[p_targets].detach()).mean()
        else:
            loss = torch.zeros(1).cuda()
        return loss

def badnets(size, a=1.):
    pattern_x, pattern_y = 2, 8
    mask = np.zeros([size, size, 3])
    mask[pattern_x:pattern_y, pattern_x:pattern_y, :] = 1 * a

    np.random.seed(0)
    pattern = np.random.rand(size, size, 3)
    pattern = np.round(pattern * 255.).astype(np.float32)
    return mask, pattern

data_root = '/home/zhihuat/BackdoorImplementations/BackdoorBox/datasets'

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10, self).__init__(root,
                                      train=train,
                                      transform=transform,
                                      target_transform=target_transform,
                                      download=download)

        self.pmark = np.zeros(len(self.targets))

    def __getitem__(self, index):
        img, target, pmark = self.data[index], self.targets[index], self.pmark[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, pmark

class PoisonedCIFAR10(object):
    def __init__(self, batch_size, num_workers, target=1, poison_rate=0.01, trigger=badnets):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
 
        self.num_classes = 10
        self.size = 32

        self.poison_rate = poison_rate
        self.mask, self.pattern = trigger(self.size)

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(self.size, 2),
            transforms.ToTensor(),
        ])
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])

    def mask_mask(self, mask_rate):
        mask_flatten = copy.deepcopy(self.mask)[..., 0:1].reshape(-1)
        maks_temp = mask_flatten[mask_flatten != 0]
        maks_mask = np.random.permutation(maks_temp.shape[0])[:int(maks_temp.shape[0] * mask_rate)]
        maks_temp[maks_mask] = 0
        mask_flatten[mask_flatten != 0] = maks_temp
        mask_flatten = mask_flatten.reshape(self.mask[..., 0:1].shape)
        mask = np.repeat(mask_flatten, 3, axis=-1)
        return mask

    def loader(self, split='train', transform=None, target_transform=None, shuffle=False, poison_rate=0., mask_rate=0., cover_rate=0., exclude_targets=None):
        train = (split == 'train')
        dataset = CIFAR10(
            root=data_root, train=train, download=True,
            transform=transform, target_transform=target_transform)

        if exclude_targets is not None:
            dataset.data = dataset.data[np.array(dataset.targets) != exclude_targets]
            dataset.targets = list(np.array(dataset.targets)[np.array(dataset.targets) != exclude_targets])

        np.random.seed(0)
        poison_index = np.random.permutation(len(dataset))[:int(len(dataset) * poison_rate)]
        n = int(len(poison_index) * cover_rate)
        poison_index, cover_index = poison_index[n:], poison_index[:n]
        for i in poison_index:
            mask = self.mask
            pattern = self.pattern
            dataset.data[i] = dataset.data[i] * (1 - mask) + pattern * mask
            dataset.targets[i] = self.target
            dataset.pmark[i] = 1
        for i in cover_index:
            mask = self.mask_mask(mask_rate)
            dataset.data[i] = dataset.data[i] * (1 - mask) + self.pattern * mask
            dataset.targets[i] = dataset.targets[i]
            dataset.pmark[i] = 2

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return dataloader, poison_index

    def get_loader(self, pr=0.02, cr=0.5, mr=0.2):
        trainloader_poison_no_cover,_ = self.loader('train', self.transform_train, poison_rate=0.5 * pr, mask_rate=0., cover_rate=0.,)
        trainloader_poison_cover, _ = self.loader('train', self.transform_train, shuffle=True, poison_rate=pr, mask_rate=mr, cover_rate=cr)
        testloader, _ = self.loader('test', self.transform_test, poison_rate=0.)
        testloader_attack, _ = self.loader('test', self.transform_test, poison_rate=1., mask_rate=0., cover_rate=0.)
        testloader_cover, _ = self.loader('test', self.transform_test, poison_rate=1., mask_rate=mr, cover_rate=1.)

        return trainloader_poison_no_cover, trainloader_poison_cover, testloader, testloader_attack, testloader_cover


class AddGTSRBTrigger:
    """Add watermarked trigger to CIFAR10 image.

    Args:
        mask (np.array): shape (H, W, C), [0 - 255]
        pattern (np.array): shape (H, W, C)
        noise (False | bool): used for convered data.
        mask_rate (0.2 | float): used for convered data.
        
    """

    def __init__(self, mask, pattern, noise=False, mask_rate=0.2):
        super(AddGTSRBTrigger, self).__init__()

        # mask, pattern = badnets(32) # 0-255, (size, size, 3)
        pattern = pattern.transpose((2, 0, 1))
        self.pattern = torch.from_numpy(pattern).int()
        
        mask = mask.transpose((2, 0, 1))
        self.ori_mask = self.mask = torch.from_numpy(mask).float()
        
        
        self.noise = noise
        self.mask_rate = mask_rate
        # self.res = self.mask * self.pattern
        # self.weight = 1.0 - self.mask
        
        
    def __call__(self, img):
        if self.noise:
            mask = self.mask_mask(self.mask_rate)
            self.mask = mask
            self.res = self.mask * self.pattern
            self.weight = 1.0 - self.mask
        else:
            self.res = self.ori_mask * self.pattern
            self.weight = 1.0 - self.ori_mask
            
        img = F.pil_to_tensor(img)
        img = self.add_trigger(img)
        img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img
    
    def add_trigger(self, img):
        
        return (self.weight * img + self.res).type(torch.uint8)
    
    def mask_mask(self, mask_rate):
        mask_flatten = copy.deepcopy(self.ori_mask)[0, ...].reshape(-1)
        maks_temp = mask_flatten[mask_flatten != 0]
        maks_mask = np.random.permutation(maks_temp.shape[0])[:int(maks_temp.shape[0] * mask_rate)]
        maks_temp[maks_mask] = 0
        mask_flatten[mask_flatten != 0] = maks_temp
        mask_flatten = mask_flatten.reshape(self.mask[0, ...].shape)
        mask = torch.repeat_interleave(mask_flatten.unsqueeze(0), 3, axis=0)
        return mask

                
class GTSRB(torchvision.datasets.GTSRB):
    def __init__(self,
                 root, 
                 split, 
                 transform, 
                 target_transform,
                 tf_insert, # 4 for train dataset, 3 for test dataset
                 tf_insert_tgt, # default 0
                 y_target,
                 poison_rate,
                 cover_rate,
                 mask_rate):
        super(GTSRB, self).__init__(root, split, transform, target_transform, download=True)
        
        total_num = len(self._samples)
        np.random.seed(0)
        poison_index = np.random.permutation(total_num)[:int(total_num * poison_rate)]
        n = int(len(poison_index) * cover_rate)
        self.poison_index, self.cover_index = poison_index[n:], poison_index[:n]
        
        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
            self.covered_transform = Compose([]) # add noise
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)
            self.covered_transform = copy.deepcopy(self.transform) # add noise
        
        mask, pattern = badnets(32)
        self.poisoned_transform.transforms.insert(tf_insert, AddGTSRBTrigger(mask, pattern, False))
        self.covered_transform.transforms.insert(tf_insert, AddGTSRBTrigger(mask, pattern, True, mask_rate))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)
        self.poisoned_target_transform.transforms.insert(tf_insert_tgt, ModifyTarget(y_target))

    def __getitem__(self, index):
        # img, target = self.data[index], int(self.targets[index])
        path, target = self._samples[index]
        
        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = PIL.Image.open(path).convert("RGB")

        # poison data
        if index in self.poison_index:
            pmark = 1
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        
        # Covered data
        elif index in self.cover_index:
            pmark = 2
            img = self.covered_transform(img)
            target = self.target_transform(target) if self.target_transform is not None else target
        
        # clean data
        else:
            pmark = 0
            img = self.transform(img) if self.transform is not None else img
            target = self.target_transform(target) if self.target_transform is not None else target

        return img, target, pmark

class PoisonedGTSRB(object):
    def __init__(self, root, batch_size, num_workers, target=1):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target = target
 
        self.num_classes = 43
        self.size = 32

        # self.poison_rate = poison_rate
        # self.mask, self.pattern = trigger(self.size)

        self.transform_train = Compose([
            ToTensor(),
            RandomHorizontalFlip(),
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            ToTensor()
        ])
        self.transform_test = Compose([
            ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            ToTensor()

        ])

        # self.transform_train = transforms.Compose([
        #     transforms.RandomCrop(self.size, 2),
        #     transforms.ToTensor(),
        # ])
        # self.transform_test = transforms.Compose([
        #     transforms.ToTensor(),
        # ])
        
    def loader(self, split, transform=None, target_transform=None, tf_insert=4, tf_insert_tgt=0,
               shuffle=False, poison_rate=0., mask_rate=0., cover_rate=0.):
        
        dataset = GTSRB(self.root, split, transform, target_transform, tf_insert, tf_insert_tgt, 
                        y_target=self.target, poison_rate=poison_rate, cover_rate=cover_rate, mask_rate=mask_rate)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)
        return dataloader
        

    def get_loader(self, dataset, pr=0.02, cr=0.5, mr=0.2):
        if dataset == 'clean':
            trainloader  = self.loader('train', self.transform_train, shuffle=True, poison_rate=0.)
            testloader = self.loader('test', self.transform_test, poison_rate=0.)
            return trainloader, testloader
        
        elif dataset == 'poison':
            trainloader_poison_no_cover  = self.loader('train', self.transform_train, tf_insert=4,  shuffle=True, poison_rate=0.5 * pr, mask_rate=0., cover_rate=0.,)
            testloader = self.loader('test', self.transform_test, poison_rate=0.)
            testloader_poison = self.loader('test', self.transform_test, tf_insert=3, poison_rate=1., mask_rate=0., cover_rate=0.)
            return trainloader_poison_no_cover, testloader, testloader_poison
        
        elif dataset == 'cover':
            trainloader_poison_cover = self.loader('train', self.transform_train, tf_insert=4, shuffle=True, poison_rate=pr, mask_rate=mr, cover_rate=cr)
            testloader = self.loader('test', self.transform_test, poison_rate=0.)
            testloader_poison = self.loader('test', self.transform_test, tf_insert=3, poison_rate=1., mask_rate=0., cover_rate=0.)
            testloader_cover = self.loader('test', self.transform_test, tf_insert=3, poison_rate=1., mask_rate=mr, cover_rate=1.)
            return trainloader_poison_cover, testloader, testloader_poison, testloader_cover


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def test(model, dataloader):
    ce_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        model.eval()

        predict_digits = []
        labels = []
        losses = []
        for batch in dataloader:
            batch_img, batch_label, _ = batch
            batch_img = batch_img.cuda()
            batch_label = batch_label.cuda()
            batch_img = model(batch_img)[0]
            loss = ce_loss(batch_img, batch_label)

            predict_digits.append(batch_img.cpu()) 
            labels.append(batch_label.cpu()) 
            if loss.ndim == 0: # scalar
                loss = torch.tensor([loss])
            losses.append(loss.cpu())
        predict_digits = torch.cat(predict_digits, dim=0) 
        labels = torch.cat(labels, dim=0) # (N)
        losses = torch.cat(losses, dim=0) # (N)
        return predict_digits, labels, losses.mean().item()
