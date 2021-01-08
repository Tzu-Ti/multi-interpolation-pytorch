#!/bin/bash

python train.py \
--save_dir checkpoints/ucf_cslstm_20210108 \
--gen_frm_dir results/ucf_cslstm_20210108 \
--log_dir ../tensorflow/logs/ucf_cslstm_20210108 \
--model_name BiLSTM \
--seq_length 11 \
--img_size 256 \
--img_channel 3 \
--num_hidden 32,32,32,32 \
--patch_size 4 \