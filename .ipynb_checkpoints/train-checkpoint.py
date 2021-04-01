__author__ = 'Titi'


import videodataset
from model.model_factory import Model
from model.model_factory_CA import Model_CA
from model.model_factory_fine_tune import Model_fine_tune
from utils.config import process_command
from utils.metrics import init_meters, init_loss

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
    if args.CA:
        LSTM = Model_CA(args, device)
    elif args.fine_tune:
        LSTM = Model_fine_tune(args, device)
    else:
        LSTM = Model(args, device)
    LSTM.network.train()
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
    writer = SummaryWriter(args.log_dir)
    
    # if args.resume, it will resume training
    if args.resume:
        if os.path.isfile(args.resume):
            print('resume!!!')
            checkpoint = torch.load(args.resume)
            model_state_dict = checkpoint['model_state_dict']
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            pretrained_epoch = checkpoint['epoch']
            mask_probability = checkpoint['mask_probability']
            # model loading weight
            LSTM.load_checkpoint(model_state_dict, optimizer_state_dict)
        else:
            raise "No this file"
            
    # Start training
    for epoch in range(pretrained_epoch, args.epochs):
        
        # Create mask to cover the even images
        # Before half epochs, it will cover all even images
        half_epochs = args.epochs // 2
        
        if epoch < half_epochs:
            mask_probability -= delta
        else:
            mask_probability = 0
        if args.LSTM_pretrained:
            mask_probability = 0
        
        # Create train and validation dataset
        train_dataset = videodataset.VideoDataset(train_video_list, args.dataset_name,
                                                  seq_length=args.seq_length,
                                                  img_size=args.img_size,
                                                  img_channel=args.img_channel,
                                                  mode="train",
                                                  mask_probability=mask_probability)

        valid_dataset = videodataset.VideoDataset(valid_video_list, args.dataset_name,
                                                  seq_length=args.seq_length, 
                                                  img_size=args.img_size, 
                                                  img_channel=args.img_channel, 
                                                  mode="valid",
                                                  mask_probability=0)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

        # training
        train_list_length = len(train_video_list)
        
        # specify loss type
        loss_type = args.loss.split('+') 
        loss_dict = init_loss(loss_type)
        
        # start training
        idx = 1
        for vid_path, seq, seq_gt in train_loader:
            print("Epoch: {}, iteration: {}/{}".format(epoch, idx, train_list_length))
            
            loss_dict = LSTM.train(seq, seq_gt, loss_dict)
            idx += args.batch_size
            
        # Write in TensorBoard
        for l in loss_type:
            writer.add_scalar('Train/{}-Loss'.format(l), np.mean(all_loss_dict[l]), epoch)
        writer.add_scalar('Train/Loss', np.mean(all_loss_dict['all_loss']), epoch)
        
        # validation
        if epoch % args.test_interval == 0:
            print("Testing...")
            all_psnr = []
            all_ssim = []
            for vid_path, seq, seq_gt, seq_origin in tqdm(valid_loader):
                batch_psnr, batch_ssim = LSTM.test(vid_path, args.gen_frm_dir, seq, seq_origin, epoch)
                for psnr, ssim in zip(batch_psnr, batch_ssim):
                    all_psnr.append(psnr)
                    all_ssim.append(ssim)
            aver_psnr = np.mean(all_psnr, axis=0)
            aver_ssim = np.mean(all_ssim, axis=0)
            
            print("Average PSNR: {}".format(aver_psnr))
            print("Average SSIM: {}".format(aver_ssim))

            # Write in TensorBoard
            psnr = np.mean(aver_psnr[1:args.seq_length-1:2])
            ssim = np.mean(aver_ssim[1:args.seq_length-1:2])
            writer.add_scalar('Train/PSNR', psnr, epoch)
            writer.add_scalar('Train/SSIM', ssim, epoch)
        
        # saving model
        if epoch % args.checkpoint_interval == 0:
            print("Saving checkpoint...")
            LSTM.save_checkpoint(epoch, mask_probability, args.save_dir)
            print("Saving model...")
            LSTM.save_model(epoch, args.save_dir)
            
    print("Saving last checkpoint...")
    LSTM.save_checkpoint(epoch, mask_probability, args.save_dir)
    LSTM.save_model(epoch, args.save_dir)
    
    print("Saving last validation...")
    all_psnr = []
    all_ssim = []
    for vid_path, seq, seq_gt, seq_origin in tqdm(valid_loader):
        batch_psnr, batch_ssim = LSTM.test(vid_path, args.gen_frm_dir, seq, seq_origin, epoch)
        for psnr, ssim in zip(batch_psnr, batch_ssim):
            all_psnr.append(psnr)
            all_ssim.append(ssim)
    aver_psnr = np.mean(all_psnr, axis=0)
    aver_ssim = np.mean(all_ssim, axis=0)
    print("Average PSNR: {}".format(aver_psnr))
    print("Average SSIM: {}".format(aver_ssim))
    
    # Write in TensorBoard
    psnr = np.mean(aver_psnr[1:args.seq_length-1:2])
    ssim = np.mean(aver_ssim[1:args.seq_length-1:2])
    writer.add_scalar('Train/PSNR', psnr, epoch)
    writer.add_scalar('Train/SSIM', ssim, epoch)
    
if __name__ == '__main__':
    main()