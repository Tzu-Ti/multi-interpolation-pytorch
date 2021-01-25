#!/bin/bash

python train.py \
--dataset_name vimeo90K \
--train_data_paths ../vimeo_septuplet/sep_trainlist.txt \
--valid_data_paths ../vimeo_septuplet/seq_vallist.txt \
--save_dir ../MVFI_output/checkpoints/vimeo_cslstm_20210125 \
--gen_frm_dir ../MVFI_output/results/vimeo_cslstm_20210125 \
--log_dir ../tensorflow/logs/vimeo_cslstm_20210125 \
--model_name BiLSTM \
--seq_length 7 \
--img_size 224 \
--img_channel 3 \
--num_hidden 32,32,32,32 \
--batch_size 32 \
--patch_size 4 \

CUDA_VISIBLE_DEVICES=0,1 python train.py --dataset_name vimeo90K --train_data_paths ../vimeo_septuplet/sep_trainlist.txt --valid_data_paths ../vimeo_septuplet/seq_vallist.txt --save_dir ../MVFI_output/checkpoints/vimeo_cslstm_20210125 --gen_frm_dir ../MVFI_output/results/vimeo_cslstm_20210125 --log_dir ../tensorflow/logs/vimeo_cslstm_20210125 --model_name BiLSTM --seq_length 7 --img_size 224 --img_channel 3 --num_hidden 32,32,32,32 --batch_size 16 --patch_size 4