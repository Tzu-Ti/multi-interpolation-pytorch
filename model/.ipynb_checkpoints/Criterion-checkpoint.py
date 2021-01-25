__author__ = 'Titi'

import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()
        
        
    def forward(self, pred_tensor, gt_tensor):
        l1 = self.l1loss(pred_tensor, gt_tensor)
        l2 = self.l2loss(pred_tensor, gt_tensor)
        
        loss = l1 + l2
        
        return loss, l1, l2
            
        
        
        