__author__ = 'Titi'

from skimage.metrics import structural_similarity as ssim

import numpy as np
import math

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

def init_loss():
    return AverageMeter()
        
def compare_PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def compare_SSIM(img1, img2):
    
    SSIM = ssim(img1, img2, data_range=255, multichannel=True)
    
    return SSIM

def quantize(img, rgb_range=255):
    return img.mul(255 / rgb_range).clamp(0, 255).round()
    
def calc_metrics(img1, img2):
    q_img1 = quantize(img1, rgb_range=1.)
    q_img2 = quantize(img2, rgb_range=1.)
    
    q_img1 = q_img1.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
    q_img2 = q_img2.cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)
    
    psnr = compare_PSNR(q_img1, q_img2)
    ssim = compare_SSIM(q_img1, q_img2)
    return psnr, ssim, q_img1, q_img2