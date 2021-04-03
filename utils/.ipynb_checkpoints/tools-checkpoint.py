__author__ = 'Titi'

from utils.pytorch_msssim import ssim_matlab as ssim_pth

import numpy as np
import math
import torch
import cv2

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def init_meters():
    psnrs = AverageMeter()
    ssims = AverageMeter()
    return psnrs, ssims

def init_loss(loss_type):
    loss_dict = {
        'all_loss': AverageMeter()
    }
    for l_type in loss_type:
        loss_dict[l_type] = AverageMeter()
    return loss_dict

def calc_psnr(img1, img2):
    '''
        Here we assume quantized(0-255) arguments.
    '''
    diff = (img1 - img2).div(255)
    mse = diff.pow(2).mean() + 1e-8
    
    return -10 * math.log10(mse)

def quantize(img, rgb_range=255):
    return img.mul(255 / rgb_range).clamp(0, 255).round()
    
def calc_metrics(img1, img2):
    q_img1 = quantize(img1, rgb_range=1.)
    q_img2 = quantize(img2, rgb_range=1.)
    
    psnr = calc_psnr(q_img1, q_img2)
    ssim = ssim_pth(q_img1.unsqueeze(0), q_img2.unsqueeze(0), val_range=255)
    
    return psnr, ssim, q_img1, q_img2

def save_image(img, path, color_mode='RGB'):
    img = img.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
    if color_mode == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif color_mode == 'BGR':
        pass
    else:
        raise
    cv2.imwrite(path, img)
