import torch
import numpy as np

class GetDataMask:
    def __init__(self, data, attack):
        if data == 'mnist':
            self.size = 28
        elif data == 'cifar10':
            self.size = 32
        
        self.attack = attack
    def get_mask(self, ori_data=None, poison_data=None):
        if self.attack == 'badnet':
            pattern_x, pattern_y = 2, 8
            mask = np.zeros([self.size, self.size, 3])
            mask[pattern_x:pattern_y, pattern_x:pattern_y, :] = 1
            mask = torch.from_numpy(mask)
        elif self.attack == 'wanet':
            assert ori_data != None and poison_data != None, 'data can not be none for WaNet attack.'
            none_zero = torch.nonzero(poison_data - ori_data, as_tuple=True)
            mask = torch.zeros_like(ori_data)
            mask[none_zero] = 1
        return mask
            