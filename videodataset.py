import cv2
import os
import imageio
import numpy as np
import random
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

def load_vimeo_data(vid_path, img_size, out_num, img_channel, mode):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
    ])
    
    T = transforms.ToTensor() # range [0, 255] -> [0.0,1.0]

    # get image from folder
    seq = [None for n in range(out_num)]
    seed = random.randint(0, 2**32)
    
    # random start index
    start_index = random.randint(1, 30-out_num)
    for t in range(out_num):
        img = Image.open(os.path.join(vid_path, "{}.png".format(t+start_index)))

        # if training, random horizontal or vertical flip
        if mode == 'train':
            torch.manual_seed(seed)
            img = transform(img)
        
        if img.size[0] != img_size:
            img = transforms.Resize((img_size, img_size))(img)
        seq[t] = T(img).numpy()
        
    # reshape back to four dimension
    seq = np.array(seq)
    
    return seq
        

def load_video_data(vid_path, img_size, out_num, img_channel, mode, start_idx):
    """
    :param vid_path: video path
    :param img_size: image size
    :param out_num: how many images extracted from video
    :param img_channel: image channel, RGB=3, GRAY=1
    :return: 'out_num' images
    """
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
    ])
    
    T = transforms.ToTensor() # range [0, 255] -> [0.0,1.0]

    vid = imageio.get_reader(vid_path, "ffmpeg")  # load video
    vid_length = vid.count_frames() # get video length
    
    if start_idx is None:
        start_idx = random.randint(0, vid_length-out_num-1) # random choose the start index
    
    # extract image from video
    seq = [None for n in range(out_num)]
    seed = random.randint(0, 2**32)
    for t in range(out_num):
        img = vid.get_data(start_idx + t)
        img = Image.fromarray(img) # convert to PIL type

        # if training, random horizontal flip
        if mode == 'train':
            torch.manual_seed(seed)
            img = transoform(img)
            
        if img.size[0] != img_size:
            img = transforms.Resize((img_size, img_size))(img)
        seq[t] = T(img).numpy()

    # reshape back to four dimension
    # [output_number, image_size, image_size, image_channel]
    seq = np.array(seq)
    
    return seq

def sample_random_noise(size):
    return np.random.uniform(low=0.0, high=1.0, size=size)

class VideoDataset(Dataset):
    def __init__(self, video_list, dataset_name, seq_length, img_size, img_channel, mode, mask_probability):
        """
        :param video_list: train or validation data paths list
        :param batch_size: mini batch size
        :param out_num: how many images extracted from video
        :param img_size: image size
        :param img_channel: image channel, RGB=3, GRAY=1
        :param mode: train or test
        """
        self.video_list = video_list
        self.dataset_name = dataset_name
        self.seq_length = seq_length
        self.img_size = img_size
        self.img_channel = img_channel
        self.mode = mode
        self.mask_probability = mask_probability
            
    def __getitem__(self, index):
        lst = self.video_list[index].split(' ')
        vid_path = lst[0]
        
        start_index = None
        # Specify video start index
        if len(lst) != 1:
            start_index = int(lst[1].split('-')[0]) - 1

        seq = None
        if self.dataset_name == 'base_dataset':
            seq = load_video_data("../"+vid_path, self.img_size, self.seq_length, self.img_channel, self.mode, start_index)
        elif self.dataset_name == 'vimeo90K':
            seq = load_vimeo_data(vid_path, self.img_size, self.seq_length, self.img_channel, self.mode)
        
        pred_length = self.seq_length // 2
        random_token = np.array([random.random() for i in range(pred_length)])
        mask_token = random_token < self.mask_probability # True to origin, False to random noise
        
        seq_gt = seq.copy()
        
        for idx, token in enumerate(mask_token):
            if not token:
                seq[idx*2+1] = sample_random_noise((self.img_channel, self.img_size, self.img_size))
        
        return vid_path, seq, seq_gt
    
    def __len__(self):
        return len(self.video_list)

