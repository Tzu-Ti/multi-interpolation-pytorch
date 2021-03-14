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
from utils.metrics import compare_PSNR, compare_SSIM
from utils.load_checkpoint import loading
from utils.pixelShuffle_torch import pixel_shuffle, seq_pixel_shuffle

class Model(object):
    def __init__(self, parser_params, device):
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
            'Bi-LSTM3': BiLSTM3.RNN,
        }

        if parser_params.model_name in networks_map:
            if parser_params.LSTM_pretrained:
                print("Loading LSTM pretrained model...")
                self.network = torch.load(parser_params.LSTM_pretrained)
                #### freeze weight
                for param in self.network.parameters():
                    param.requires_grad = False
            else:
                Network = networks_map[parser_params.model_name]
                self.network = Network(self.num_layers, num_hidden,
                                       parser_params.seq_length, parser_params.patch_size, parser_params.batch_size,
                                       parser_params.img_size, parser_params.img_channel,
                                       parser_params.filter_size, parser_params.stride)
                self.network = DataParallel(self.network, device_ids = [0, 1, 2, 3])
                
            self.network.to(device)
        else:
            raise ValueError('Name of network unknown {}'.format(parser_params.model_name))

        self.optimizer = Adam(self.network.parameters(), lr=parser_params.lr)
        
        # specify loss type
        loss_type = parser_params.loss.split('+')
        self.criterion = Criterion.Loss(loss_type)
        self.pred_loss = 0
            
    def train(self, input_tensor, gt_tensor):
        patch_tensor = seq_pixel_shuffle(input_tensor, 1 / self.patch_size).type(torch.cuda.FloatTensor)
        patch_rev_tensor = torch.flip(patch_tensor, (1, ))

        self.optimizer.zero_grad()
        pred_seq = self.network(patch_tensor, patch_rev_tensor)

        loss_value = self.criterion(pred_seq, gt_tensor.type(torch.cuda.FloatTensor))
        
        if self.pred_loss == 0:
            self.pred_loss = loss_value['all_loss'].data.item()
        
        loss_value['all_loss'].backward()
        
        # if loss exploding, skip this training
        if loss_value['all_loss'].data.item() > 1.5 * self.pred_loss:
            print("[Warning] Loss exploding...( {} )".format(loss_value['all_loss'].detach().cpu().numpy()))
            return loss_value
        
        self.optimizer.step()
        self.pred_loss = loss_value['all_loss'].data.item()
        
        print("Loss: {}".format(loss_value['all_loss'].detach().cpu().numpy()))
        
        return loss_value
        
    def test(self, vid_path, gen_frm_dir, input_tensor, gt_tensor, epoch):
        patch_tensor = seq_pixel_shuffle(input_tensor, 1 / self.patch_size).type(torch.cuda.FloatTensor)
        patch_rev_tensor = torch.flip(patch_tensor, (1, ))
        
        pred_seq = self.network(patch_tensor, patch_rev_tensor)
        pred_seq = pred_seq.detach().cpu().numpy()

        pred_seq = pred_seq * 255
        gt_tensor = gt_tensor.numpy() * 255

        batch_psnr = []
        batch_ssim = []
        for batch in range(self.batch_size):
            # get file path and name
            try:
                path = vid_path[batch]
            except:
                continue
            f_name = path.split('/')[-1]
            
            ep_folder = os.path.join(gen_frm_dir, str(epoch))
            f_folder = os.path.join(ep_folder, f_name)
#             if not os.path.isdir(f_folder):
#                 os.makedirs(f_folder)
            
            batch_pred_seq = pred_seq[batch]
            batch_gt_seq = gt_tensor[batch]
            
            seq_psnr = []
            seq_ssim = []
            for t in range(self.seq_length):
                pred_img = np.transpose(batch_pred_seq[t], (1, 2, 0))
                gt_img = np.transpose(batch_gt_seq[t], (1, 2, 0))
                
                gt_size = np.shape(gt_img)
                
                pred_img = cv2.resize(pred_img, (gt_size[1], gt_size[0]))
                
                # save prediction and GT
#                 pred_path = os.path.join(f_folder, "pd-{}.png".format(t+1))
#                 cv2.imwrite(pred_path, pred_img)
#                 gt_path = os.path.join(f_folder, "gt-{}.png".format(t+1))
#                 cv2.imwrite(gt_path, gt_img)
                
                psnr = compare_PSNR(pred_img, gt_img)
                ssim = compare_SSIM(pred_img, gt_img)
                seq_psnr.append(psnr)
                seq_ssim.append(ssim)
                
            batch_psnr.append(seq_psnr) 
            batch_ssim.append(seq_ssim)
#             print("PSNR: {}, SSIM: {}".format(seq_psnr, seq_ssim))
        return batch_psnr, batch_ssim
    
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
        
    def save_model(self, epoch, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, 'model_{}.pt'.format(epoch))
        torch.save(self.network.module.state_dict(), save_path)