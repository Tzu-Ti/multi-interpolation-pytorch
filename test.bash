#!/bin/bash

python test.py --dataset vimeo90K --valid_data_paths videolist/Vimeo90K/test_list.txt --gen_frm_dir ../MVFI_output/results/test/ --model BiLSTM --seq_length 7 --img_size 224 --img_channel 3 --num_hidden 32,32,32,32 --batch_size 64 --checkpoint_path ../checkpoint_199.tar