__author__ = 'Titi'

from skimage.metrics import structural_similarity as ssim

import numpy as np
import math
        
def compare_PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def compare_SSIM(img1, img2):
    
    SSIM = ssim(img1, img2, data_range=255, multichannel=True)
    
    return SSIM
    