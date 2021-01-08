__author__ = 'Titi'

import torch
import torch.nn as nn
import numpy as np
import cv2
import os

from torch.optim import Adam
from torch.nn import DataParallel

from model import BiLSTM, BiLSTM3
from model import Criterion
from utils.metrics import compare_PSNR
from utils.pixelShuffle import pixelDownShuffle, pixelUpShuffle

class Model(object):
    def __init__(self, parser_params):
        """
        :param parser_params: parser args
        """
        
        num_hidden = [int(x) for x in parser_params.num_hidden.split(',')]
        
        self.seq_length = parser_params.seq_length
        self.batch_size = parser_params.batch_size
        self.patch_size = parser_params.patch_size
        self.num_layers = len(num_hidden)
        networks_map = {
            'BiLSTM': BiLSTM.RNN,
            'BiLSTM-3': BiLSTM3.RNN
        }

        if parser_params.model_name in networks_map:
            Network = networks_map[parser_params.model_name]
            self.network = Network(self.num_layers, num_hidden,
                                   parser_params.seq_length, parser_params.patch_size, parser_params.batch_size,
                                   parser_params.img_size, parser_params.img_channel,
                                   parser_params.filter_size, parser_params.stride).cuda(0)
#             self.network = DataParallel(self.network)
        else:
            raise ValueError('Name of network unknown {}'.format(parser_params.model_name))
            
        self.optimizer = Adam(self.network.parameters(), lr=parser_params.lr)
#         self.optimizer = DataParallel(self.optimizer)
        self.criterion = Criterion.Loss()
            
    def train(self, input_tensor, gt_tensor):
        patch_tensor = pixelDownShuffle(input_tensor, self.patch_size).cuda(0)
        patch_rev_tensor = torch.flip(patch_tensor, (1, ))
        patch_gt_tensor = pixelDownShuffle(gt_tensor, self.patch_size).cuda(0)
        
        self.optimizer.zero_grad()
        pred_seq = self.network(patch_tensor, patch_rev_tensor)

        loss, l1_loss, l2_loss = self.criterion(pred_seq, patch_gt_tensor)
        
        loss.backward()
        self.optimizer.step()
        
        print("Loss: {}".format(loss.detach().cpu().numpy()))
        
        return loss.detach().cpu().numpy(), l1_loss.detach().cpu().numpy(), l2_loss.detach().cpu().numpy()
        
    def test(self, vid_path, gen_frm_dir, input_tensor, gt_tensor, epoch):
        patch_tensor = pixelDownShuffle(input_tensor, self.patch_size).cuda(0)
        patch_rev_tensor = torch.flip(patch_tensor, (1, ))
        
        pred_seq = self.network(patch_tensor, patch_rev_tensor)
        
        pred_seq = pixelUpShuffle(pred_seq.detach().cpu().numpy(), self.patch_size)

        pred_seq = pred_seq * 255
        gt_tensor = gt_tensor.numpy() * 255

        batch_psnr = []
        batch_ssim = []
        for batch in range(self.batch_size):
            # get file path and name
            path = vid_path[batch]
            print(path)
            f_name = path.split('/')[-1]
            
            ep_folder = os.path.join(gen_frm_dir, str(epoch))
            f_folder = os.path.join(ep_folder, f_name)
            if not os.path.isdir(f_folder):
                os.makedirs(f_folder)
            
            batch_pred_seq = pred_seq[batch]
            batch_gt_seq = gt_tensor[batch]
            
            seq_psnr = []
            seq_ssim = []
            for t in range(self.seq_length):
                pred_img = np.transpose(batch_pred_seq[t], (1, 2, 0))
                gt_img = np.transpose(batch_gt_seq[t], (1, 2, 0))
                
                pred_path = os.path.join(f_folder, "pd-{}.png".format(t+1))
                cv2.imwrite(pred_path, pred_img)
                gt_path = os.path.join(f_folder, "gt-{}.png".format(t+1))
                cv2.imwrite(gt_path, gt_img)
                
                psnr = compare_PSNR(pred_img, gt_img)
                seq_psnr.append(psnr)
                
            batch_psnr.append(seq_psnr) 
            print("PSNR: {}".format(seq_psnr))
        return batch_psnr
    
    def save_checkpoint(self, epoch, mask_probability, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        save_path = os.path.join(save_dir, 'checkpoint_{}.tar'.format(epoch))
        torch.save({
            'epoch': epoch,
            'mask_probability': mask_probability,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)
        
    def load_checkpoint(self, model_state_dict, optimizer_state_dict):
        self.network.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)