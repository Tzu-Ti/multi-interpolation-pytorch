#!/bin/bash

python train.py \
--save_dir checkpoints/ucf_cslstm_20201206_1 \
--gen_frm_dir results/ucf_cslstm_20201206_1 \
--log_dir ../tensorflow/logs/ucf_cslstm_20201206_1 \
--model_name BiLSTM \
--seq_length 11 \
--img_size 128 \
--img_channel 3 \
--num_hidden 64,64,64,64 \
--patch_size 4 \