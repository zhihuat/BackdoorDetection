import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from core.attacks.BadNet_BELT import badnets
import matplotlib.pyplot as plt


EPSILON =  1e-07
class Pixel:
    def __init__(self, model, dataloader, size, lr):
        self.model = model
        self.dataloader = dataloader
        self.clip_max = 1.0
        self.device = 'cuda'
        
        self.asr_bound = 0.9
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5
        
        self.init_cost = 1e-3
        self.cost = self.init_cost
        
        
        # initialize patterns with random values
        for i in range(2):
            init_pattern = torch.rand([3, size, size]).cuda()
            if i == 0:
                self.pattern_pos_tanh = torch.arctanh((init_pattern - 0.5)/ (2. - EPSILON))
                self.pattern_pos_tanh.requires_grad = True
            else:
                self.pattern_neg_tanh = torch.arctanh((init_pattern - 0.5)/ (2. - EPSILON))
                self.pattern_neg_tanh.requires_grad = True
                
        self.radius = torch.tensor(0.2).cuda()
        
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.nll_loss_fn = nn.NLLLoss()
        self.opt = optim.Adam([self.pattern_pos_tensor, self.pattern_neg_tensor], lr=lr, betas=(0.9, 0.999))
                
        self.num_noise_vec = 4
        self.inflate = 2
        
    def detect(self, target_label, epochs=10, verbose=True):
        pattern_best     = torch.zeros(self.pattern_shape).to(self.device)
        pattern_pos_best = torch.zeros(self.pattern_shape).to(self.device)
        pattern_neg_best = torch.zeros(self.pattern_shape).to(self.device)
        reg_best = float('inf')
        pixel_best  = float('inf')
        
        # hyper-parameters to dynamically adjust loss weight
        cost_up_counter   = 0
        cost_down_counter = 0
        
        best_loss = np.inf
        best_pattern = None
        best_pixel = None
        for epoch in range(epochs):
            acc_list, cls_loss_list = [], []
            reg_loss_list = []
            used_samples = 0
            for batch in self.dataloader:
                mini_batches = get_minibatches(batch, self.num_noise_vec)
                for inputs, label in mini_batches:
                    used_samples += inputs.shape[0]
                    
                    pattern_cur, losses  = self.train(inputs, label, target_label, reg=False)
                    
                    (correct, cls_loss, reg_loss) = losses
                    acc_list.append(correct)
                    cls_loss_list.append(cls_loss.cpu().detach().item())
                    reg_loss_list.append(reg_loss.cpu().detach().item())
                    
            pixel_cur = np.count_nonzero(
                            np.sum(np.abs(pattern_cur.cpu().numpy()), axis=0)
                        )
            avg_acc = np.sum(acc_list) / used_samples        
            avg_cls_loss = np.mean(cls_loss_list)
            avg_reg_loss = np.mean(reg_loss_list)
            
            if avg_cls_loss < best_loss and pixel_cur < pixel_best:
                best_loss = avg_cls_loss
                best_pixel = pixel_cur
                best_pattern = pattern_cur.detach().cpu()
                
            # helper variables for adjusting loss weight
            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            # adjust loss weight
            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if self.cost == 0:
                    self.cost = self.init_cost
                else:
                    self.cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                self.cost /= self.cost_multiplier_down
                
            if verbose:
                print(f'Epoch {epoch}, Accuracy : {avg_acc}, Pixel Number: {best_pixel}, Cls Loss: {avg_cls_loss}, Reg Loss: {avg_reg_loss}')
                invert_trigger_array = (best_pattern.squeeze().detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                plt.imsave(f'plots/trigger_invert_epoch{epoch}.png', invert_trigger_array)
        return best_pattern
                
    def train(self, inputs, label, target_label, reg=True):        

        inputs = inputs.cuda()
        label = label.cuda()
        self.opt.zero_grad()
        
        pattern_pos = ((torch.tanh(self.pattern_pos_tanh) / (2. - EPSILON) + 0.5))
        pattern_neg = -((torch.tanh(self.pattern_neg_tanh) / (2. - EPSILON) + 0.5))
        
        
        ### cls loss of trigger
        poison_data = torch.clamp(inputs + pattern_pos + pattern_neg, min=0.0, max=self.clip_max)
        poison_label = torch.ones(poison_data.shape[0], device='cuda', dtype=torch.long) * target_label
        logits = self.model(poison_data)[0]
        
        
        ### losses
        cls_loss = self.cls_loss_fn(logits, poison_label)
        
        reg_pos  = torch.max(torch.tanh(self.pattern_pos_tanh / 10)\
                            / (2 - EPSILON) + 0.5, axis=0)[0]
        reg_neg  = torch.max(torch.tanh(self.pattern_neg_tensor / 10)\
                        / (2 - EPSILON) + 0.5, axis=0)[0]
        reg_loss = torch.sum(reg_pos) + torch.sum(reg_neg)
        
        
        loss = cls_loss  + reg_loss * self.cost    
        
        loss.backward()
        self.opt.step()
        
        correct = torch.sum(torch.eq(logits.argmax(1), poison_label)).cpu().detach().item()
        
        losses =  (correct, cls_loss, reg_loss)
        
        threshold = self.clip_max / 255.0
        pattern_pos_cur = pattern_pos.detach()
        pattern_neg_cur = pattern_neg.detach()
        pattern_pos_cur[(pattern_pos_cur < threshold)\
                            & (pattern_pos_cur > -threshold)] = 0
        pattern_neg_cur[(pattern_neg_cur < threshold)\
                            & (pattern_neg_cur > -threshold)] = 0
        pattern_cur = pattern_pos_cur + pattern_neg_cur
        pattern_cur = pattern_pos_cur + pattern_neg_cur
        
        return pattern_cur, losses
        

def get_minibatches(batch, num_batches):
    X = batch[0]
    y = batch[1]

    batch_size = len(X) // num_batches
    for i in range(num_batches):
        yield X[i*batch_size : (i+1)*batch_size], y[i*batch_size : (i+1)*batch_size]