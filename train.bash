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

python train.py --dataset vimeo90K --train_data_paths videolist/Vimeo90K/train_list.txt --valid_data_paths videolist/Vimeo90K/test_list400.txt --training_name 20210405 --model BiLSTM --seq_length 7 --img_size 256 --img_channel 3 --num_hidden 64,64,64,64 --batch_size 8 --lr 0.005 --checkpoint_path ../MVFI_output/20210405/checkpoint/checkpoint_best.tar --save_results