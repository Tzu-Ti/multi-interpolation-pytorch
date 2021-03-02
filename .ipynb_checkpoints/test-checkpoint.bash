#!/bin/bash

python test.py \
--model_name BiLSTM \
--seq_length 3 \
--img_size 224 \
--img_channel 3 \
--num_hidden 32,32,32,32 \
--batch_size 32 \
--patch_size 4 \
--checkpoint_path ../MVFI_output/checkpoints/ucf_cslstm_20210129_L3/checkpoint_90.tar \
--gen_frm_dir ../MVFI_output/results/test \
# --LSTM_pretrained model_90.pt \
# --CA_patch_size 4 \
# --n_resgroups 3 \
# --n_resblocks 6
python test.py --model_name BiLSTM --seq_length 15 --img_size 224 --img_channel 3 --num_hidden 32,32,32,32 --batch_size 10 --patch_size 4 --checkpoint_path ../MVFI_output/checkpoints/ucf_cslstm_20210208_L15/checkpoint_99.tar --gen_frm_dir ../MVFI_output/results/test