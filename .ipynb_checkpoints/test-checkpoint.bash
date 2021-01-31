#!/bin/bash

python test.py \
--model_name Bi-LSTM3 \
--seq_length 7 \
--img_size 224 \
--img_channel 3 \
--num_hidden 32,32,32 \
--batch_size 32 \
--patch_size 4 \
--checkpoint_path ../MVFI_output/checkpoints/ucf_cslstm_20210126_3_3/CA_checkpoint_99.tar \
--gen_frm_dir ../MVFI_output/results/test \
--LSTM_pretrained ../MVFI_output/checkpoints/ucf_cslstm_20210116/model_90.pt \
--CA_patch_size 4 \
--n_resgroups 3 \
--n_resblocks 3