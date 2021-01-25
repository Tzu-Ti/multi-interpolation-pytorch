__author__ = 'Titi'

import torch
import torch.nn as nn
import numpy as np

from model.CausalLSTM import CausalLSTMCell
from model.GradientHighwayUnit import GHU

from utils.pixelShuffle_torch import pixel_shuffle

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, seq_length, patch_size, batch_size, img_size, img_channel, filter_size, stride):
        super(RNN, self).__init__()

        self.frame_channel = patch_size * patch_size * img_channel
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.pred_length = seq_length//2 # frame interpolation numbers
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        
        lstm_fw = []
        lstm_bw = []

        width = img_size // patch_size

        # initialize lstm architecture
        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i-1]
            lstm_fw.append(
                CausalLSTMCell(in_channel, num_hidden[i], width, filter_size, stride)
            )
            lstm_bw.append(
                CausalLSTMCell(in_channel, num_hidden[i], width, filter_size, stride)
            )
        self.lstm_fw = nn.ModuleList(lstm_fw)
        self.lstm_bw = nn.ModuleList(lstm_bw)
        
        # initialize 2, 3, 4 layers concat architecture including hidden state and memory state
        self.hidden_concat_conv_l2 = nn.Conv2d(num_hidden[0]*2, num_hidden[0],
                                               kernel_size=1, stride=1, padding=0)
        self.mem_concat_conv_l2 = nn.Conv2d(num_hidden[0]*2, num_hidden[0],
                                            kernel_size=1, stride=1, padding=0)
        self.hidden_concat_conv_l3 = nn.Conv2d(num_hidden[1]*2, num_hidden[0],
                                               kernel_size=1, stride=1, padding=0)
        self.mem_concat_conv_l3 = nn.Conv2d(num_hidden[1]*2, num_hidden[0],
                                            kernel_size=1, stride=1, padding=0)
        self.hidden_concat_conv_l4 = nn.Conv2d(num_hidden[2]*2, num_hidden[0],
                                               kernel_size=1, stride=1, padding=0)
        self.mem_concat_conv_l4 = nn.Conv2d(num_hidden[2]*2, num_hidden[0],
                                            kernel_size=1, stride=1, padding=0)
        self.hidden_concat_conv_gen = nn.Conv2d(num_hidden[3]*2, num_hidden[0],
                                               kernel_size=1, stride=1, padding=0)
        
        # initialize generate convolution
        self.conv_last = nn.Conv2d(num_hidden[num_layers-1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0)
        
        # Initialize GHU unit
        self.ghu = GHU(in_channel, num_hidden[0], width, filter_size, stride)
        

    def forward(self, fw_seq, bw_seq):
        batch = fw_seq.shape[0]
        height = fw_seq.shape[3]
        width = fw_seq.shape[4]
        
        # initialize zero
        zero = torch.zeros([batch, self.num_hidden[0], height, width]).type(torch.cuda.FloatTensor)
        
        # Initialize LSTM hidden state and cell state
        hidden_fw = []
        hidden_bw = []
        cell_fw = []
        cell_bw = []
        
        tm_hidden_fw = [[None for i in range(self.seq_length)] for k in range(self.num_layers)]
        tm_hidden_bw = [[None for i in range(self.seq_length)] for k in range(self.num_layers)]
        tm_mem_fw = [[None for i in range(self.seq_length)] for k in range(self.num_layers)]
        tm_mem_bw = [[None for i in range(self.seq_length)] for k in range(self.num_layers)]
        
        for i in range(self.num_layers):
            hidden_fw.append(zero)
            hidden_bw.append(zero)
            cell_fw.append(zero)
            cell_bw.append(zero)
            
        memory_fw = zero
        memory_bw = zero
        z_t_fw = zero
        z_t_bw = zero
        
        gen_images = []
        
        ########## Layer 1 lstm pass ##########
        for t in range(self.seq_length):
            # forward
            inputs_fw = fw_seq[:, t]
            hidden_fw[0], cell_fw[0], memory_fw = self.lstm_fw[0](inputs_fw, hidden_fw[0], cell_fw[0], memory_fw)
            
            # GHU
            # Only layer 1 need to use GHU
            z_t_fw = self.ghu(hidden_fw[0], z_t_fw)
            
            tm_hidden_fw[0][t] = z_t_fw
            tm_mem_fw[0][t] = memory_fw
            
            # backword
            inputs_bw = bw_seq[:, t]
            hidden_bw[0], cell_bw[0], memory_bw = self.lstm_bw[0](inputs_bw, hidden_bw[0], cell_fw[0], memory_bw)
            
            z_t_bw = self.ghu(hidden_bw[0], z_t_bw)
            
            tm_hidden_bw[0][t] = z_t_bw
            tm_mem_bw[0][t] = memory_bw
            
        ########## Layer 2 ##########
        # only have 5 lstm
        hiddenConcated_l2 = [None for i in range(self.pred_length)]
        memConcated_l2 = [None for i in range(self.pred_length)]
        # layer 2 merge hidden and memory
        for t in range(self.pred_length):
            # Concatenate
            hiddenConcat = torch.cat([tm_hidden_fw[0][t*2], tm_hidden_bw[0][(self.pred_length-t-1)*2]], axis=1)
            memConcat = torch.cat([tm_mem_fw[0][t*2], tm_mem_bw[0][(self.pred_length-t-1)*2]], axis=1)
            
            # Convolution back to origin channel
            hiddenConcated_l2[t] = self.hidden_concat_conv_l2(hiddenConcat)
            memConcated_l2[t] = self.mem_concat_conv_l2(memConcat)
            
        # layer 2 lstm pass
        for t in range(self.pred_length):
            # forward
            hidden_fw[1], cell_fw[1], memory_fw = self.lstm_fw[1](hiddenConcated_l2[t], hidden_fw[1], cell_fw[1], memConcated_l2[t])
            
            tm_hidden_fw[1][t] = hidden_fw[1]
            tm_mem_fw[1][t] = memory_fw
            
            # backword
            hidden_bw[1], cell_bw[1], memory_bw = self.lstm_bw[1](hiddenConcated_l2[self.pred_length-t-1], hidden_bw[1], cell_bw[1], memConcated_l2[self.pred_length-t-1])
            
            tm_hidden_bw[1][t] = hidden_bw[1]
            tm_mem_bw[1][t] = memory_bw
            
        ########## Layer 3 ##########
        # only have 5 lstm
        hiddenConcated_l3 = [None for i in range(self.pred_length)]
        memConcated_l3 = [None for i in range(self.pred_length)]
        # layer 3 merge hidden and memory
        for t in range(self.pred_length):
            # Concatenate
            hiddenConcat = torch.cat([tm_hidden_fw[1][t], tm_hidden_bw[1][self.pred_length-t-1]], axis=1)
            memConcat = torch.cat([tm_mem_fw[1][t], tm_mem_bw[1][self.pred_length-t-1]], axis=1)
            
            # Convolution back to origin channel
            hiddenConcated_l3[t] = self.hidden_concat_conv_l3(hiddenConcat)
            memConcated_l3[t] = self.mem_concat_conv_l3(memConcat)
            
        # layer 3 lstm pass
        for t in range(self.pred_length):
            # forward
            hidden_fw[2], cell_fw[2], memory_fw = self.lstm_fw[2](hiddenConcated_l3[t], hidden_fw[2], cell_fw[2], memConcated_l3[t])
            
            tm_hidden_fw[2][t] = hidden_fw[2]
            tm_mem_fw[2][t] = memory_fw
            
            # backword
            hidden_bw[2], cell_bw[2], memory_bw = self.lstm_bw[2](hiddenConcated_l3[self.pred_length-t-1], hidden_bw[2], cell_bw[2], memConcated_l3[self.pred_length-t-1])
            
            tm_hidden_bw[2][t] = hidden_bw[2]
            tm_mem_bw[2][t] = memory_bw
            
        ########## Layer 4 ##########
        # only have 5 lstm
        hiddenConcated_l4 = [None for i in range(self.pred_length)]
        memConcated_l4 = [None for i in range(self.pred_length)]
        # layer 4 merge hidden and memory
        for t in range(self.pred_length):
            # Concatenate
            hiddenConcat = torch.cat([tm_hidden_fw[2][t], tm_hidden_bw[2][self.pred_length-t-1]], axis=1)
            memConcat = torch.cat([tm_mem_fw[2][t], tm_mem_bw[2][self.pred_length-t-1]], axis=1)
            
            # Convolution back to origin channel
            hiddenConcated_l4[t] = self.hidden_concat_conv_l4(hiddenConcat)
            memConcated_l4[t] = self.mem_concat_conv_l4(memConcat)
            
        # layer 4 lstm pass
        for t in range(self.pred_length):
            # forward
            hidden_fw[3], cell_fw[3], memory_fw = self.lstm_fw[3](hiddenConcated_l3[t], hidden_fw[3], cell_fw[3], memConcated_l3[t])
            
            tm_hidden_fw[3][t] = hidden_fw[3]
            tm_mem_fw[3][t] = memory_fw
            
            # backword
            hidden_bw[3], cell_bw[3], memory_bw = self.lstm_bw[3](hiddenConcated_l4[self.pred_length-t-1], hidden_bw[3], cell_bw[3], memConcated_l4[self.pred_length-t-1])
            
            tm_hidden_bw[3][t] = hidden_bw[3]
            tm_mem_bw[3][t] = memory_bw
            
        
        ########## Merge prediction hidden state ##########
        hiddenConcatConv = [None for i in range(self.pred_length)]
        for t in range(self.pred_length):
            # Concatenate
            hiddenConcat = torch.cat([tm_hidden_fw[3][t], tm_hidden_bw[3][self.pred_length-t-1]], axis=1)
            
            # Convolution back to origin channel
            hiddenConcatConv[t] = self.hidden_concat_conv_gen(hiddenConcat)
        
        x_gen = [None for i in range(self.seq_length)]
        # Generate complete output
        for t in range(self.seq_length):
            if t % 2 == 0:
                gen = fw_seq[:, t]
                x_gen[t] = pixel_shuffle(gen, self.patch_size)
            else:
                gen = self.conv_last(hiddenConcatConv[t//2])
                x_gen[t] = pixel_shuffle(gen, self.patch_size)
                
        pred_frames = torch.stack(x_gen, dim=0).permute(1, 0, 2, 3, 4).contiguous()

        return pred_frames
