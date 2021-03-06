__author__ = 'Titi'


import videodataset
from model.model_factory import Model
from utils.tools import init_meters
from utils.config import process_command

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import random
import os
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
    # validation
    print("Testing...")
    psnrs, ssims = [], []
    for _ in range(args.seq_length):
        init_psnr, init_ssim = init_meters()
        psnrs.append(init_psnr)
        ssims.append(init_ssim)
        
    with torch.no_grad():
        for vid_path, seq, seq_gt in tqdm(valid_loader):
            psnrs, ssims = LSTM.test(vid_path, args.gen_frm_dir, seq, seq_gt, 0, psnrs, ssims)

    for t in range(args.seq_length):
        print("{}\tPSNR: {:0.3f}, SSIM: {:0.3f}".format(t+1, psnrs[t].avg, ssims[t].avg))
    
if __name__ == '__main__':
    main()