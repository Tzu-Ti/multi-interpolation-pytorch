__author__ = 'Titi'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def cal_gdl(predImg, target):
    """
    Gradient Difference Loss
    Image gradient difference loss as defined by Mathieu et al. (https://arxiv.org/abs/1511.05440).
    """
    alpha = 1
    
    # [batch_size, seq_length, channels, height, width]
    predImg_col_grad = torch.abs(predImg[:, :, :, :, :-1] - predImg[:, :, :, :, 1:])
    predImg_row_grad = torch.abs(predImg[:, :, :, 1:, :] - predImg[:, :, :, :-1, :])
    target_col_grad = torch.abs(target[:, :, :, :, :-1] - target[:, :, :, :, 1:])
    target_row_grad = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])
    col_grad_loss = torch.abs(predImg_col_grad - target_col_grad)
    row_grad_loss = torch.abs(predImg_row_grad - target_row_grad)
    
    #loss = col_grad_loss + row_grad_loss
    loss = torch.sum(col_grad_loss ** alpha) + torch.sum(row_grad_loss ** alpha)
    return loss

def cal_TV(predImg):
    """
    Total variation loss
    """
    TVLoss_weight = 1
    
    # [seq_length, channels, height, width]
    seq_length = predImg.size()[0]
    h_x = predImg.size()[2]
    w_x = predImg.size()[3]
    count_h = (predImg.size()[2] - 1) * predImg.size()[3]
    count_w = predImg.size()[2] * (predImg.size()[3] - 1)
    h_tv = torch.pow((predImg[:, :, 1:, :] - predImg[:, :, :h_x-1, :]), 2).sum()
    w_tv = torch.pow((predImg[:, :, :, 1:] - predImg[:, :, :, :w_x-1]), 2).sum()
    
    return TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / seq_length

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        
        # extract conv5 4 features from the VGG-19 model pretrained on ImageNet dataset 
        self.vgg = nn.Sequential(*modules[:35])
        self.vgg = nn.DataParallel(self.vgg).cuda()
        self.vgg.requires_grad = False
        
    def forward(self, predImg, target):
        def _forward(x):
            x = self.vgg(x)
            return x
        vgg_pred = _forward(predImg)
        with torch.no_grad():
            vgg_target = _forward(target.detach())
        loss = F.mse_loss(vgg_pred, vgg_target)
        
        return loss
        
        
    
    
class Loss(nn.Module):
    def __init__(self, loss_type):
        super(Loss, self).__init__()
        
        self.loss_function = {}
        if 'L1' in loss_type:
            self.loss_function['L1'] = nn.L1Loss()
        if 'L2' in loss_type:
            self.loss_function['L2'] = nn.MSELoss()
        if 'vgg' in loss_type:
            self.loss_function['vgg'] = VGG()
            
        self.loss_type = loss_type
        
        
    def forward(self, pred_tensor, gt_tensor):  
        loss_value = {
            'all_loss': 0
        }

        for l in self.loss_type:
            loss_value[l] = self.loss_function[l](pred_tensor, gt_tensor)
            loss_value['all_loss'] += loss_value[l]
            
        
        return loss_value
            
class FineTuneLoss(nn.Module):
    def __init__(self):
        super(FineTuneLoss, self).__init__()
        
        self.vggloss = VGG()
        
    def forward(self, pred_tensor, gt_tensor):
        batch_size = pred_tensor.size()[0]
        vgg = 0
        for b in range(batch_size):
            vgg += self.vggloss(pred_tensor[b], gt_tensor[b])
            
        loss = vgg / batch_size
        
        return loss
        