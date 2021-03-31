__author__ = 'Titi'


import videodataset
from model.model_factory import Model
from model.model_factory_CA import Model_CA

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import random
import os
import cv2

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def process_command():
    parser = argparse.ArgumentParser()
    
    # data I/O
    parser.add_argument('--dataset_name', default='base_dataset',
                        help='The name of dataset.')
    parser.add_argument('--train_data_paths', default='videolist/UCF-101/train_data_list.txt',
                        help='train data paths.')
    parser.add_argument('--valid_data_paths', default='videolist/UCF-101/val_data_list_sample320.txt',
                        help='validation data paths.')
    parser.add_argument('--gen_frm_dir',
                        required=True,
                        help='directory to store result.')
    
    # model
    parser.add_argument('--model_name',
                        required=True,
                        help='Model name')
    parser.add_argument('--seq_length',
                        required=True,
                        type=int,
                        help='sequence length') 
    parser.add_argument('--img_size',
                        required=True,
                        type=int,
                        help='image size')
    parser.add_argument('--img_channel',
                        required=True,
                        type=int,
                        help='image channel')
    parser.add_argument('--num_hidden', default='64,64,64,64',
                        help='Hidden state channel')
    parser.add_argument('--filter_size', default=5,
                        type=int,
                        help='The filter size of convolution in the lstm')
    parser.add_argument('--stride', default=1,
                        type=int,
                        help='The stride of convolution in the lstm')
    parser.add_argument('--patch_size', default=4,
                        type=int,
                        help='PixelShuffle parameter')
    parser.add_argument('--LSTM_pretrained', default='',
                        help="LSTM pretrained model path")
    parser.add_argument('--CA_patch_size', default=8,
                        type=int,
                        help="Channel attention pixelshuffle factor")
    parser.add_argument('--n_resgroups', default=5,
                        type=int,
                        help="Channel attention number of Residual Groups")
    parser.add_argument('--n_resblocks', default=12,
                        type=int,
                        help="Channel attention number of Residual Blocks")
    
    # training setting
    parser.add_argument('--lr', default=0.001,
                        type=float,
                        help='base learning rate')
    parser.add_argument('--batch_size', default=4,
                        type=int,
                        help='batch size for training.')
    parser.add_argument('--epochs', default=100,
                        type=int,
                        help='batch size for training.')
    parser.add_argument('--checkpoint_interval', default=10,
                        type=int,
                        help='number of epoch to save model parameter')
    parser.add_argument('--test_interval', default=10,
                        type=int,
                        help='number of epoch to test')
    parser.add_argument('--checkpoint_path', required=True,
                        help='checkpoint path')
    parser.add_argument('--loss', default='L1+L2',
                        help='ex. []+[] (L1, L2, vgg)')
    
    return parser.parse_args()

def main():
    args = process_command()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Loading LSTM model
    print("Loading LSTM model...", end='')
    
    LSTM = Model(args, device)
    LSTM.network.eval()
    print("...OK")
    
    # get video list from video_list_paths
    with open(args.valid_data_paths, 'r') as f:
        valid_video_list = [line.strip() for line in f.readlines()]
    
    # Loading model parameter
    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        pretrained_epoch = checkpoint['epoch']
        mask_probability = checkpoint['mask_probability']
        # model loading weight
        LSTM.load_checkpoint(model_state_dict, optimizer_state_dict)
    else:
        raise "[Error] No this checkpoint file"
        
    # Create validation dataset
    valid_dataset = videodataset.VideoDataset(valid_video_list, args.dataset_name,
                                              seq_length=args.seq_length, 
                                              img_size=args.img_size, 
                                              img_channel=args.img_channel, 
                                              mode="valid",
                                              mask_probability=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        
    # validation
    print("Testing...")
    all_psnr = []
    all_ssim = []
    for vid_path, seq, seq_gt in valid_loader:
        batch_psnr, batch_ssim = LSTM.test(vid_path, args.gen_frm_dir, seq, seq_gt, 0)
        for psnr, ssim in zip(batch_psnr, batch_ssim):
            all_psnr.append(psnr)
            all_ssim.append(ssim)
    aver_psnr = np.mean(all_psnr, axis=0)
    aver_ssim = np.mean(all_ssim, axis=0)
    print("Average PSNR: {}".format(aver_psnr))
    print("Average SSIM: {}".format(aver_ssim))
    
if __name__ == '__main__':
    main()