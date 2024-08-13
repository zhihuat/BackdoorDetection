import torch
from torch import nn
from torch import optim

import numpy as np


EPSILON =  1e-07
class NeuralCleanse:
    def __init__(self, model, dataloader, size, lr):
        self.model = model
        self.dataloader = dataloader
        
        mask = torch.rand(size, size).cuda()
        self.mask_tanh = torch.arctanh((mask - 0.5)/ (2. - EPSILON))
        self.mask_tanh.requires_grad = True

        pattern = torch.rand([1, 3, size, size], dtype=torch.float32).cuda() * 255.0
        self.pattern_tanh = torch.arctanh((pattern / 255.0 - 0.5) * (2 - EPSILON))
        self.pattern_tanh.requires_grad = True
        
        
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.opt = optim.Adam([self.mask_tanh, self.pattern_tanh], lr=lr, betas=(0.9, 0.999))
        
    def detect(self, target_label, epochs=10, verbose=True):
        best_acc = 0
        best_loss = np.inf
        best_pattern = None
        best_mask = None
        for epoch in range(epochs):
            acc_list, cls_loss_list = [], []
            reg_loss_list = []
            used_samples = 0
            for (data, label) in self.dataloader:
                data = data.cuda()
                label = label.cuda()
                self.opt.zero_grad()
                
                poison_label = torch.ones_like(label, device='cuda') * target_label
                mask_raw = ((torch.tanh(self.mask_tanh) / (2. - EPSILON) + 0.5)).repeat(3, 1, 1).unsqueeze(0)
                pattern_raw = ((torch.tanh(self.pattern_tanh) / (2. - EPSILON) + 0.5) )
            
                poison_data = mask_raw * pattern_raw + (1 - mask_raw) * data
                logits = self.model(poison_data)[0]
                cls_loss = self.cls_loss_fn(logits, poison_label)
                reg_loss = torch.sum(torch.abs(mask_raw)) / 3
                
                loss = cls_loss + reg_loss *  1e-2
                
                used_samples += data.shape[0]
                correct = torch.sum(torch.eq(logits.argmax(1), poison_label)).cpu().detach().item()
                acc_list.append(correct)
                cls_loss_list.append(cls_loss.cpu().detach().item())
                reg_loss_list.append(reg_loss.cpu().detach().item())
                
                loss.backward()
                self.opt.step()
            
            avg_acc = np.sum(acc_list) / used_samples
            avg_cls_loss = np.mean(cls_loss_list)
            avg_reg_loss = np.mean(reg_loss_list)
            if avg_cls_loss < best_loss and avg_acc > best_acc:
                best_loss = avg_cls_loss
                best_acc = avg_acc
                best_pattern = pattern_raw.detach().cpu()
                best_mask = mask_raw.detach().cpu( )
            if verbose:
                print(f'Epoch {epoch}, Accuracy : {avg_acc}, Cls Loss: {avg_cls_loss}, Reg Loss: {avg_reg_loss}')
        return best_pattern, best_mask[0, 0, :, :]
                

from sam import SAM
class NeuralCleanseSAM:
    def __init__(self, model, dataloader, size, lr):
        self.model = model
        self.dataloader = dataloader
        
        mask = torch.rand(size, size).cuda()
        self.mask_tanh = torch.arctanh((mask - 0.5)/ (2. - EPSILON))
        self.mask_tanh.requires_grad = True

        pattern = torch.rand([1, 3, size, size], dtype=torch.float32).cuda() * 255.0
        self.pattern_tanh = torch.arctanh((pattern / 255.0 - 0.5) * (2 - EPSILON))
        self.pattern_tanh.requires_grad = True
        
        
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.opt = SAM([self.mask_tanh, self.pattern_tanh], torch.optim.Adam , lr=0.1)
        
    def detect(self, target_label, epochs=10, verbose=True):
        best_acc = 0
        best_loss = np.inf
        best_pattern = None
        best_mask = None
        for epoch in range(epochs):
            acc_list, cls_loss_list = [], []
            reg_loss_list = []
            used_samples = 0
            for (data, label) in self.dataloader:
                data = data.cuda()
                label = label.cuda()
                self.opt.zero_grad()
                
                poison_label = torch.ones_like(label, device='cuda') * target_label
                mask_raw = ((torch.tanh(self.mask_tanh) / (2. - EPSILON) + 0.5)).repeat(3, 1, 1).unsqueeze(0)
                pattern_raw = ((torch.tanh(self.pattern_tanh) / (2. - EPSILON) + 0.5) )
            
                poison_data = mask_raw * pattern_raw + (1 - mask_raw) * data
                logits = self.model(poison_data)[0]
                cls_loss = self.cls_loss_fn(logits, poison_label)
                reg_loss = torch.sum(torch.abs(mask_raw)) / 3
                
                loss = cls_loss + reg_loss *  1e-3 
                
                used_samples += data.shape[0]
                correct = torch.sum(torch.eq(logits.argmax(1), poison_label)).cpu().detach().item()
                acc_list.append(correct)
                cls_loss_list.append(cls_loss.cpu().detach().item())
                reg_loss_list.append(reg_loss.cpu().detach().item())
                
                loss.backward()
                self.opt.first_step(zero_grad=True)
                
                ####################
                mask_raw = ((torch.tanh(self.mask_tanh) / (2. - EPSILON) + 0.5)).repeat(3, 1, 1).unsqueeze(0)
                pattern_raw = ((torch.tanh(self.pattern_tanh) / (2. - EPSILON) + 0.5) )
            
                poison_data = mask_raw * pattern_raw + (1 - mask_raw) * data
                logits = self.model(poison_data)[0]
                cls_loss = self.cls_loss_fn(logits, poison_label)
                reg_loss = torch.sum(torch.abs(mask_raw)) / 3
                
                loss = cls_loss + reg_loss *  1e-3 
                loss.backward()
                self.opt.second_step(zero_grad=True)
                ##########################
            
            avg_acc = np.sum(acc_list) / used_samples
            avg_cls_loss = np.mean(cls_loss_list)
            avg_reg_loss = np.mean(reg_loss_list)
            if avg_cls_loss < best_loss and avg_acc > best_acc:
                best_loss = avg_cls_loss
                best_acc = avg_acc
                best_pattern = pattern_raw.detach().cpu()
                best_mask = mask_raw.detach().cpu( )
            if verbose:
                print(f'Epoch {epoch}, Accuracy : {avg_acc}, Cls Loss: {avg_cls_loss}, Reg Loss: {avg_reg_loss}')
        return best_pattern, best_mask[0, 0, :, :]
                

class NeuralCleanseWithMask:
    def __init__(self, model, dataloader, size, lr):
        self.model = model
        self.dataloader = dataloader
        
        pattern = torch.rand([1, 3, size, size], dtype=torch.float32).cuda() * 255.0
        self.pattern_tanh = torch.arctanh((pattern / 255.0 - 0.5) * (2 - EPSILON))
        self.pattern_tanh.requires_grad = True
        
        
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.opt = optim.Adam([self.pattern_tanh], lr=lr, betas=(0.9, 0.999))
        
    def detect(self, target_label, mask, epochs=10, verbose=True):
        best_acc = 0
        best_loss = np.inf
        best_pattern = None
        # best_mask = None
        if len(mask.size()) == 2:
            mask_raw = mask.repeat(3, 1, 1).unsqueeze(0).cuda()
        else:
            mask_raw = mask.cuda()
        
        for epoch in range(epochs):
            acc_list, cls_loss_list = [], []
            # reg_loss_list = []
            used_samples = 0
            for (data, label) in self.dataloader:
                data = data.cuda()
                label = label.cuda()
                self.opt.zero_grad()
                
                poison_label = torch.ones_like(label, device='cuda') * target_label
                # mask_raw = ((torch.tanh(self.mask_tanh) / (2. - EPSILON) + 0.5)).repeat(3, 1, 1).unsqueeze(0)
                pattern_raw = ((torch.tanh(self.pattern_tanh) / (2. - EPSILON) + 0.5) )
            
                poison_data = mask_raw * pattern_raw + (1 - mask_raw) * data
                logits = self.model(poison_data)[0]
                cls_loss = self.cls_loss_fn(logits, poison_label)
                # reg_loss = torch.sum(torch.abs(mask_raw)) / 3
                
                loss = cls_loss
                
                used_samples += data.shape[0]
                correct = torch.sum(torch.eq(logits.argmax(1), poison_label)).cpu().detach().item()
                acc_list.append(correct)
                cls_loss_list.append(cls_loss.cpu().detach().item())
                # reg_loss_list.append(reg_loss.cpu().detach().item())
                
                loss.backward()
                self.opt.step()
            
            avg_acc = np.sum(acc_list) / used_samples
            avg_cls_loss = np.mean(cls_loss_list)
            # avg_reg_loss = np.mean(reg_loss_list)
            if avg_cls_loss < best_loss and avg_acc > best_acc:
                best_loss = avg_cls_loss
                best_acc = avg_acc
                best_pattern = pattern_raw.detach().cpu()
                # best_mask = mask_raw.detach().cpu( )
            if verbose:
                print(f'Epoch {epoch}, Accuracy : {avg_acc}, Cls Loss: {avg_cls_loss}')
        return best_pattern
                