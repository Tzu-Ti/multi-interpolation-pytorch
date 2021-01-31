#!/bin/bash

python train.py \
--train_data_paths videolist/UCF-101/train_data_list.txt \
--valid_data_paths videolist/UCF-101/val_data_list_sample320.txt \
--save_dir ../MVFI_output/checkpoints/ucf_cslstm_20210130_1_12 \
--gen_frm_dir ../MVFI_output/results/ucf_cslstm_20210130_1_12 \
--log_dir ../tensorflow/logs/ucf_cslstm_20210130_1_12 \
--model_name Bi-LSTM3 \
--seq_length 7 \
--img_size 224 \
--img_channel 3 \
--num_hidden 32,32,32 \
--batch_size 32 \
--patch_size 4 \
--LSTM_pretrained ../MVFI_output/checkpoints/ucf_cslstm_20210116/model_90.pt \
--CA_patch_size 8 \
--n_resgroups 1 \
--n_resblocks 12