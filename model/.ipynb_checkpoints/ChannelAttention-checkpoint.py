__author__ = 'Titi'

import torch
import torch.nn as nn
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, input_channel):
        super(ResBlock, self).__init__()
        reflection_padding = 3 // 2
        
        self.channel = input_channel
        self.head = nn.Sequential(
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1),
            nn.ReLU(True),
            nn.ReflectionPad2d(reflection_padding),
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1)
        )
        
        # CA layer
        self.CA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channel, self.channel // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channel // 8, self.channel, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
#         print("x shape {}".format(np.shape(x)))
        tm_head = self.head(x)
        print(tm_head)
        raise
#         print("tm_head shape {}".format(np.shape(tm_head)))
        CA_weight = self.CA(tm_head)
#         print("CA_weight shape {}".format(np.shape(CA_weight)))
        CA_return = tm_head * CA_weight
        Block_return = x + CA_return
        
        return Block_return
        

class ResModule(nn.Module):
    def __init__(self, n_blocks, input_channel):
        super(ResModule, self).__init__()
        self.n_blocks = n_blocks
        self.input_channel = input_channel
        
        blocks = [ResBlock(self.input_channel) for _ in range(self.n_blocks)]
        self.blocks = nn.Sequential(*blocks)
        
    def forward(self, x):
        y = x
        y = self.blocks(y)
            
        y += x
        return y