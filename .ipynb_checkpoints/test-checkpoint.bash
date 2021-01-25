#!/bin/bash

python test.py \
--model_name BiLSTM-3 \
--seq_length 11 \
--img_size 256 \
--img_channel 3 \
--num_hidden 32,32,32 \
--patch_size 8 \
--checkpoint_path checkpoints/ucf_3layer_patch8_cslstm_20210108/checkpoint_80.tar \
--gen_frm_dir results/test