__author__ = 'Titi'


import videodataset
from model.model_factory import Model

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
    parser.add_argument('--valid_data_paths', default='videolist/UCF-101/val_data_list_sample300.txt',
                        help='validation data paths.')
    parser.add_argument('--save_dir',
                        required=True,
                        help='directory to store trained checkpoint.')
    parser.add_argument('--gen_frm_dir',
                        required=True,
                        help='directory to store result.')
    parser.add_argument('--log_dir',
                        required=True,
                        help='log directory for TensorBoard')
    
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
    parser.add_argument('--resume', default='',
                        help='checkpoint path')
    
    return parser.parse_args()

def main():
    args = process_command()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Loading LSTM model
    print("Loading LSTM model..")
    LSTM = Model(args, device)
    
    # get video list from video_list_paths
    with open(args.train_data_paths, 'r') as f:
        train_video_list = [line.strip() for line in f.readlines()]
        
    with open(args.valid_data_paths, 'r') as f:
        valid_video_list = [line.strip() for line in f.readlines()]
        
    # Training setting
    mask_probability = 1
    delta = 0.035
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
            
    # Start training
    for epoch in range(pretrained_epoch, args.epochs):
        
        # Create mask to cover the even images
        # Before half epochs, it will cover all even images
        half_epochs = args.epochs // 2
        
        if epoch < half_epochs:
            mask_probability -= delta
        else:
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

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        # training
        train_list_length = len(train_video_list)
        all_loss, all_l1_loss, all_l2_loss = [], [], []
        idx = 1
        for vid_path, seq, seq_gt in train_loader:
            print("Epoch: {}, iteration: {}/{}".format(epoch, idx, train_list_length))
            
            loss, l1_loss, l2_loss = LSTM.train(seq, seq_gt)
            all_loss.append(loss)
            all_l1_loss.append(l1_loss)
            all_l2_loss.append(l2_loss)

            idx += args.batch_size
            
        # Write in TensorBoard
        writer.add_scalar('Train/L1-Loss', np.mean(all_l1_loss), epoch)
        writer.add_scalar('Train/L2-Loss', np.mean(all_l2_loss), epoch)
        writer.add_scalar('Train/Loss', np.mean(all_loss), epoch)
        
        # validation
        if epoch % args.test_interval == 0:
            print("Testing...")
            all_psnr = []
            for vid_path, seq, seq_gt in valid_loader:
                batch_psnr = LSTM.test(vid_path, args.gen_frm_dir, seq, seq_gt, epoch)
                for psnr in batch_psnr:
                    all_psnr.append(psnr)
            aver_psnr = np.mean(all_psnr, axis=0)
            print("Average PSNR: {}".format(aver_psnr))
            # Write in TensorBoard
            writer.add_scalar('Train/PSNR', aver_psnr)
        
        # saving model
        if epoch % args.checkpoint_interval == 0:
            print("Saving checkpoint...")
            LSTM.save_checkpoint(epoch, mask_probability, args.save_dir)
            
    print("Saving last checkpoint...")
    LSTM.save_checkpoint(epoch, mask_probability, args.save_dir)
    
    print("Saving last validation...")
    all_psnr = []
    for vid_path, seq, seq_gt in valid_loader:
        batch_psnr = LSTM.test(vid_path, args.gen_frm_dir, seq, seq_gt, epoch)
        for psnr in batch_psnr:
            all_psnr.append(psnr)
    aver_psnr = np.mean(all_psnr, axis=0)
    print("Average PSNR: {}".format(aver_psnr))
    
if __name__ == '__main__':
    main()