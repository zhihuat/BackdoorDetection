import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from core.attacks.BadNet_BELT import badnets
import matplotlib.pyplot as plt

EPSILON =  1e-07
class NeuralDiffClean:
    def __init__(self, model, dataloader, size, lr):
        self.model = model
        self.dataloader = dataloader
        
        # mask, pattern = badnets(32)
        
        mask = torch.rand(size, size).cuda()
        mask[2:8, 2:8] = 1
        # mask = torch.from_numpy(mask[:, :, 0]).cuda().float()
        self.mask_tanh = torch.arctanh((mask - 0.5)/ (2. - EPSILON))
        self.mask_tanh.requires_grad = True

        pattern = torch.rand([1, 3, size, size], dtype=torch.float32).cuda() * 255.0
        # pattern = torch.from_numpy(pattern).permute([2, 0, 1]).unsqueeze(0).cuda().float()
        self.pattern_tanh = torch.arctanh((pattern / 255.0 - 0.5) * (2 - EPSILON))
        self.pattern_tanh.requires_grad = True
        
        self.radius = torch.tensor(0.5).cuda()
        
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.kl_loss_fn = nn.KLDivLoss()
        self.opt = optim.Adam([self.mask_tanh, self.pattern_tanh], lr=lr, betas=(0.9, 0.999))
        
        self.num_noise_vec = 10
        
    def detect(self, target_label, epochs=10, verbose=True):
        best_acc = 0
        best_loss = np.inf
        best_pattern = None
        best_mask = None
        for epoch in range(epochs):
            acc_list, cls_loss_list = [], []
            noise_acc_list, noise_cls_loss_list = [], []
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
                reg_loss = torch.sum(torch.abs(mask_raw)) / 3  # 3 channels
                
                straw_label = self.model(data)[0].argmax(1).detach()
                
                neigh_cls_loss = torch.zeros(1).cuda()
                for j in range(4):
                    gaussian_noise = torch.randn_like(data)  # .clamp(-data, 1-data)
                    noise_data = (poison_data + mask_raw * self.radius * gaussian_noise).clamp(0, 1)
                    noise_logits = self.model(noise_data)[0]
                    neigh_cls_loss += self.cls_loss_fn(noise_logits, straw_label) / 4
                
                loss = neigh_cls_loss # + reg_loss * 1e-2 cls_loss + 
                
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
                print(f'Epoch {epoch}, Accuracy : {avg_acc}, Cls Loss: {avg_cls_loss}, Reg Loss: {avg_reg_loss}, Neigh Cls Loss: {neigh_cls_loss}')
        return best_pattern, best_mask[0, 0, :, :]


class NeuralDiffClean_v2:
    def __init__(self, model, dataloader, size, lr):
        self.model = model
        self.dataloader = dataloader
        
        # mask, pattern = badnets(32)
        mask = torch.rand(size, size).cuda()
        mask[2:8, 2:8] = 1
        # mask = torch.from_numpy(mask[:, :, 0]).cuda().float()
        self.mask_tanh = torch.arctanh((mask - 0.5)/ (2. - EPSILON))
        self.mask_tanh.requires_grad = True

        pattern = torch.rand([1, 3, size, size], dtype=torch.float32).cuda() * 255.0
        # pattern = torch.from_numpy(pattern).permute([2, 0, 1]).unsqueeze(0).cuda().float()
        self.pattern_tanh = torch.arctanh((pattern / 255.0 - 0.5) * (2 - EPSILON))
        self.pattern_tanh.requires_grad = True
        
        mask_raw = ((torch.tanh(self.mask_tanh) / (2. - EPSILON) + 0.5)).repeat(3, 1, 1).unsqueeze(0)
        pattern_raw = ((torch.tanh(self.pattern_tanh) / (2. - EPSILON) + 0.5) )
        invert_trigger_array = ((mask_raw * pattern_raw)[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        plt.imsave(f'plots/trigger_invert_init.png', invert_trigger_array)
        
        self.radius = torch.tensor(0.2).cuda()
        
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.nll_loss_fn = nn.NLLLoss()
        self.kl_loss_fn = nn.KLDivLoss()
        self.opt = optim.Adam([self.mask_tanh, self.pattern_tanh], lr=lr, betas=(0.9, 0.999))
        
        self.num_noise_vec = 4
        self.inflate = 2
        
    def detect(self, target_label, epochs=10, verbose=True):
        best_loss = np.inf
        best_noise_loss = np.inf
        best_reg_loss = np.inf
        best_pattern = None
        best_mask = None
        for epoch in range(epochs):
            acc_list, cls_loss_list = [], []
            noise_acc_list, noise_cls_loss_list = [], []
            reg_loss_list = []
            used_samples = 0
            for batch in self.dataloader:
                mini_batches = get_minibatches(batch, self.num_noise_vec)
                for inputs, label in mini_batches:
                    used_samples += inputs.shape[0]
                    if epoch < 5:
                        ### warm up
                        pattern_raw, mask_raw, losses  = self.train(epoch, inputs, label, target_label, reg=False)
                    else:
                        pattern_raw, mask_raw, losses  = self.train(epoch, inputs, label, target_label, reg=True)
                    
                    (correct, noise_correct, cls_loss, reg_loss,noise_cls_loss) = losses
                    acc_list.append(correct)
                    noise_acc_list.append(noise_correct)
                    cls_loss_list.append(cls_loss.cpu().detach().item())
                    reg_loss_list.append(reg_loss.cpu().detach().item())
                    noise_cls_loss_list.append(noise_cls_loss.cpu().detach().item())
                    
            avg_acc = np.sum(acc_list) / used_samples
            noise_avg_acc = np.sum(noise_acc_list) / used_samples
            avg_cls_loss = np.mean(cls_loss_list)
            avg_reg_loss = np.mean(reg_loss_list)
            avg_noise_cls_loss = np.mean(noise_cls_loss_list)
            
            # if avg_cls_loss < best_loss and avg_noise_cls_loss < best_noise_loss:
            if reg_loss < best_reg_loss:
                best_loss = avg_cls_loss
                best_noise_loss = avg_noise_cls_loss
                best_reg_loss = reg_loss
                best_pattern = pattern_raw.detach().cpu()
                best_mask = mask_raw.detach().cpu( )
            if verbose:
                print(f'Epoch {epoch}, Accuracy : {avg_acc}, Noise Accuracy: {noise_avg_acc}, Cls Loss: {avg_cls_loss}, Reg Loss: {avg_reg_loss}, Neigh Cls Loss: {avg_noise_cls_loss}')
                invert_trigger_array = ((best_pattern * best_mask)[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                plt.imsave(f'plots/trigger_invert_epoch{epoch}.png', invert_trigger_array)
        return best_pattern, best_mask[0, 0, :, :]
                
    def train(self, epoch, inputs, label, target_label, reg=True):        

        inputs = inputs.cuda()
        label = label.cuda()
        self.opt.zero_grad()

        mask_raw = ((torch.tanh(self.mask_tanh / 1) / (2. - EPSILON) + 0.5)).repeat(3, 1, 1).unsqueeze(0)
        pattern_raw = ((torch.tanh(self.pattern_tanh / 4) / (2. - EPSILON) + 0.5))
        
        ### cls loss of trigger
        poison_data = mask_raw * pattern_raw + (1 - mask_raw) * inputs
        poison_label = torch.ones(poison_data.shape[0], device='cuda', dtype=torch.long) * target_label
        logits = self.model(poison_data)[0]
        
        ### cls loss of trigger + noise
        poison_data_repreat = poison_data.repeat((1, self.num_noise_vec*self.inflate, 1, 1)).reshape(-1, *(poison_data.shape)[1:])
        noise = torch.randn_like(poison_data_repreat, device='cuda')
        
        noise_data = (poison_data_repreat + mask_raw * self.radius * noise).clamp(0, 1)
        noise_logits = self.model(noise_data)[0]
        noise_prob = F.softmax(noise_logits, dim=1).reshape(-1, self.num_noise_vec*self.inflate, logits.shape[-1]).mean(1)
        noise_logprob = torch.log(noise_prob.clamp(min=1e-20))
        straw_label = self.model(inputs)[0].argmax(1).detach()
        
        ### losses
        cls_loss = self.cls_loss_fn(logits, poison_label)
        reg_loss = torch.sum(torch.abs(mask_raw)) / 3
        noise_cls_loss = self.nll_loss_fn(noise_logprob, straw_label)
        if reg:
            loss = cls_loss + noise_cls_loss + reg_loss * 1e-2
        else:
            loss = cls_loss + noise_cls_loss
        
        loss.backward()
        self.opt.step()
        
        correct = torch.sum(torch.eq(logits.argmax(1), poison_label)).cpu().detach().item()
        noise_correct = torch.sum(torch.eq(noise_prob.argmax(1), straw_label)).cpu().detach().item()
        losses =  (correct, noise_correct, cls_loss, reg_loss,noise_cls_loss)
        
        return pattern_raw, mask_raw, losses
        

def get_minibatches(batch, num_batches):
    X = batch[0]
    y = batch[1]

    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]