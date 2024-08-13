import torch
from torch import nn
from torch import optim

import numpy as np


EPSILON =  1e-07
class RobustRadius:
  def __init__(self, model, data, label, mask):
    self.model = model
    self.data = data
    self.mask = mask
    self.clean_label = label
        
    self.radius = torch.tensor(1).cuda()
        
  def search(self, target_label, num_noise=100, verbose=False):
    num_batches = num_noise // 1000
    
    min_radius = 0
    max_radius = 10
    radius = 1
    while np.abs(min_radius - max_radius) >= 0.00001:
      correct = 0
      clean_acc = 0
      radius = min_radius + (max_radius - min_radius) / 2

      for i in range(num_batches):
        noise = torch.randn(1000, 3, 32, 32).clamp(-self.data, 1-self.data)
        inputs = (self.data + radius * noise * self.mask)
        inputs = inputs.clamp(0, 1)
        inputs = inputs.cuda().float()
        preds = self.model(inputs)[0].argmax(1)
        
        correct += preds.eq(target_label).sum().cpu().item()
        correct_clean += preds.eq(self.clean_label).sum().cpu().item()

      acc = correct/num_noise
      clean_acc = correct_clean/num_noise

      
      if verbose:
        print(f'acc: {acc}, clean acc: {clean_acc}, min_radisu: {min_radius}, max_radius: {max_radius}, radius:{radius}')
        
      if acc < 1:
        max_radius = radius
      elif acc == 1:
        min_radius = radius
    return radius
        
        
        
        
class ExclusiveRadius:
  def __init__(self, model, data, label, mask):
    self.model = model
    self.data = data
    self.mask = mask
    self.clean_label = label
        
    # self.radius = torch.tensor(1).cuda()

        
  def search(self, target_label, num_noise=100, verbose=True):
    num_batches = num_noise // 1000
    min_radius = 0.
    max_radius = 100
    radius = 1
    while np.abs(min_radius - max_radius) >= 0.001:
      correct = 0
      correct_clean = 0
      radius = min_radius + (max_radius - min_radius) / 2

      for i in range(num_batches):
        noise = torch.randn(1000, 3, 32, 32).clamp(-self.data, 1-self.data)
        inputs = (self.data + radius * noise * self.mask)
        inputs = inputs.clamp(0, 1)
        inputs = inputs.cuda().float()
        preds = self.model(inputs)[0].argmax(1)
        
        correct += preds.eq(target_label).sum().cpu().item()
        correct_clean += preds.eq(self.clean_label).sum().cpu().item()

      acc = correct/num_noise
      clean_acc = correct_clean/num_noise
      
      if verbose:
        print(f'acc: {acc}, clean acc: {clean_acc}, min_radisu: {min_radius}, max_radius: {max_radius}, radius:{radius}')
        
      if acc > 0:
        min_radius = radius
      elif acc == 0:
        max_radius = radius
    return radius
        
        
class RandomizedSmoothRadius:
  def __init__(self, model, data, label, mask):
    self.model = model
    self.data = data
    self.mask = mask
    self.clean_label = label
        
    # self.radius = torch.tensor(1).cuda()

        
  def search(self, target_label, num_noise=100, verbose=False):
    num_batches = num_noise // 1000
    min_radius = 0.
    max_radius = 100
    radius = 1
    while np.abs(min_radius - max_radius) >= 0.001:
      correct = 0
      correct_clean = 0
      radius = min_radius + (max_radius - min_radius) / 2

      for i in range(num_batches):
        noise = torch.randn(1000, 3, 32, 32).clamp(-self.data, 1-self.data)
        inputs = (self.data + radius * noise * self.mask)
        inputs = inputs.clamp(0, 1)
        inputs = inputs.cuda().float()
        preds = self.model(inputs)[0].argmax(1)
        
        correct += preds.eq(target_label).sum().cpu().item()
        correct_clean += preds.eq(self.clean_label).sum().cpu().item()

      acc = correct/num_noise
      clean_acc = correct_clean/num_noise
      
      if verbose:
        print(f'acc: {acc}, clean acc: {clean_acc}, min_radisu: {min_radius}, max_radius: {max_radius}, radius:{radius}')
        
      if acc > 0.5:
        min_radius = radius
      elif acc < 0.5:
        max_radius = radius
    return radius