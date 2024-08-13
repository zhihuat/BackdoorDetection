import torch
import torch.nn.functional as F

from torch.linalg import matrix_norm
from torch import autograd

class SpectralNorm:
    def __init__(self, model):
        self.model = model
        self.pattern = pattern
        
    def get_loss(self, data, pattern):
        poison_data = mask * pattern + (1 - mask) * data
        logit = F.softmax(self.model(poison_data)[0], dim=1)[:, 1].mean()
        gradients = autograd.grad(outputs=logit, inputs=pattern, grad_outputs=torch.ones_like(logit),
                                  create_graph=True, retain_graph=True)[0]
        gradients_norm = matrix_norm(gradients, 2).pow(2).sum().sqrt()
        return gradients_norm
        
        