#!/bin/bash

python train.py \
--save_dir checkpoints/ucf_cslstm_coding \
--gen_frm_dir results/ucf_cslstm_coding \
--log_dir ../tensorflow/logs/ucf_cslstm_coding \
--model_name BiLSTM \
--seq_length 7 \
--img_size 256 \
--img_channel 3 \
--num_hidden 32,32,32 \
--patch_size 4 \

python train.py --save_dir checkpoints/ucf_cslstm_coding --gen_frm_dir results/ucf_cslstm_coding --log_dir ../tensorflow/logs/ucf_cslstm_coding --model_name BiLSTM-3 --seq_length 7 --img_size 224 --img_channel 3 --num_hidden 32,32,32 --patch_size 4