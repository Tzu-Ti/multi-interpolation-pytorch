__author__ = 'Titi'


import videodataset
from model.model_factory import Model
# from model.model_factory_CA import Model_CA
# from model.model_factory_fine_tune import Model_fine_tune
from utils.config import process_command
from utils.tools import init_meters, init_loss

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

def train(LSTM, args, train_loader, train_list_length, epoch, writer):
    # specify loss type
    loss_type = args.loss.split('+') 
    loss_dict = init_loss(loss_type)
    
    idx = 1
    for vid_path, seq, seq_gt in train_loader:
        print("Epoch: {}, iteration: {}/{}".format(epoch, idx, train_list_length))

        loss_dict = LSTM.train(seq, seq_gt, loss_dict)
        idx += args.batch_size

    # Write in TensorBoard
    for l in loss_type:
        writer.add_scalar('Train/{}-Loss'.format(l), loss_dict[l].avg, epoch)
    writer.add_scalar('Train/Loss', loss_dict['all_loss'].avg, epoch)
    
def validation(LSTM, args, valid_loader, epoch, gen_dir, writer):
    print("Testing...")
    psnrs, ssims = [], []
    for _ in range(args.seq_length):
        init_psnr, init_ssim = init_meters()
        psnrs.append(init_psnr)
        ssims.append(init_ssim)

    with torch.no_grad():
        for vid_path, seq, seq_gt in tqdm(valid_loader):
            psnrs, ssims = LSTM.test(vid_path, gen_dir, seq, seq_gt, epoch, psnrs, ssims)

    psnr_avg, ssim_avg = init_meters()
    for t in range(args.seq_length):
        print("{}\tPSNR: {:0.3f}, SSIM: {:0.3f}".format(t+1, psnrs[t].avg, ssims[t].avg))
        if t % 2 == 1:
            psnr_avg.update(psnrs[t].avg)
            ssim_avg.update(ssims[t].avg)

    # Write in TensorBoard
    writer.add_scalar('Train/PSNR', psnr_avg.avg, epoch)
    writer.add_scalar('Train/SSIM', ssim_avg.avg, epoch)
    
    return psnr_avg.avg

def main():
    args = process_command()
    
    ckpt_dir = os.path.join(args.save_dir, args.training_name, 'checkpoints') # directory to store trained checkpoint
    gen_dir = os.path.join(args.save_dir, args.training_name, 'results') # directory to store result
    log_dir = os.path.join(args.save_dir, 'logs', args.training_name) # log directory for TensorBoard
    
    print("Initialize cuDNN...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Loading LSTM model
    print("Loading LSTM model...", end='')
    LSTM = Model(args, device)
    print("...OK")
    
    # get video list from video_list_paths
    with open(args.train_data_paths, 'r') as f:
        train_video_list = [line.strip() for line in f.readlines()]
        
    with open(args.valid_data_paths, 'r') as f:
        valid_video_list = [line.strip() for line in f.readlines()]
        
    # Training setting
    mask_probability = 1
    delta = args.delta
    pretrained_epoch = 0
    writer = SummaryWriter(log_dir)
    best_psnr = 0 
    
    # if args.resume, it will resume training
    if args.checkpoint_path:
        if os.path.isfile(args.checkpoint_path):
            print('resume!!!')
            checkpoint = torch.load(args.checkpoint_path)
            model_state_dict = checkpoint['model_state_dict']
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            pretrained_epoch = checkpoint['epoch'] + 1
            mask_probability = checkpoint['mask_probability']
            best_psnr = checkpoint['best_psnr']
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
    
    # Start training
    for epoch in range(pretrained_epoch, args.epochs):
        
        # Create mask to cover the even images
        if mask_probability > 0:
            mask_probability -= delta
        else:
            mask_probability = 0
        if args.LSTM_pretrained:
            mask_probability = 0
        
        # Create train dataset
        train_dataset = videodataset.VideoDataset(train_video_list, args.dataset_name,
                                                  seq_length=args.seq_length,
                                                  img_size=args.img_size,
                                                  img_channel=args.img_channel,
                                                  mode="train",
                                                  mask_probability=mask_probability)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
        # training
        train_list_length = len(train_video_list)
        LSTM.network.train()
        train(LSTM, args, train_loader, train_list_length, epoch, writer)
        
        # validation
        LSTM.network.eval()
        psnr = validation(LSTM, args, valid_loader, epoch, gen_dir, writer)
        
        if psnr > best_psnr:
            print("Saving checkpoint...")
            best_psnr = psnr
            LSTM.save_checkpoint(epoch, mask_probability, ckpt_dir, best_psnr)
    
if __name__ == '__main__':
    main()
