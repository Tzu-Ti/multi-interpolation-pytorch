#!/bin/bash

python train.py \
--save_dir ../MVFI_output/checkpoints/ucf_cslstm_20210125_big \
--gen_frm_dir ../MVFI_output/results/ucf_cslstm_20210125_big \
--log_dir ../tensorflow/logs/ucf_cslstm_20210125_big \
--model_name Bi-LSTM3 \
--seq_length 7 \
--img_size 224 \
--img_channel 3 \
--num_hidden 32,32,32 \
--batch_size 32 \
--patch_size 4 \
--LSTM_pretrained ../MVFI_output/checkpoints/ucf_cslstm_20210116/model_90.pt