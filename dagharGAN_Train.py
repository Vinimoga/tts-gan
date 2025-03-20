#!/usr/bin/env bash

import os
import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    parser.add_argument('--t_checkpoint', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    opt = parser.parse_args()

    return opt
args = parse_args()

'''
For Each experiment change:
    -max_iter : for the maximum number of iterations
    -t_checkpoint : for the number of epochs after the model will save a checkpoint
    -class_name : for the name of the class of daghar dataset or the UniMib data
    -exp_name : for the name of the experiment that will be saaved in log, along with the date
'''

max_iter = 10000 #300 épocas: 137694 
checkpoint_number = 4
seq_len = 60
channels = 6
log_dir = 'logs/eval_different_constructions/'
path = '../DAGHAR_split_25_10_all/train/'
data_path = path + 'data/'
classes = os.listdir(data_path)
label_path = [path + 'label/' + s.replace('DAGHAR', 'Label') for s in classes]


default_string = f"CUDA_VISIBLE_DEVICES=0 python train_GAN_dahar.py -gen_bs 16 -dis_bs 16 \
                --dist-url 'tcp://localhost:4321' --dist-backend 'nccl' --world-size 1 \
                --rank {args.rank} --dataset daghar --bottom_width 8 --img_size 32 \
                --gen_model my_gen --dis_model my_dis --df_dim 384 --d_heads 4 --d_depth 3 \
                --g_depth 5,4,2 --dropout 0 --latent_dim 100 --gf_dim 1024 --num_workers 6 \
                --g_lr 0.0001 --d_lr 0.0003 --optimizer adam --loss lsgan --wd 1e-3 --beta1 0.9 \
                --beta2 0.999 --phi 1 --batch_size 64 --num_eval_imgs 50000 --init_type xavier_uniform \
                --n_critic 1 --val_freq 20 --print_freq 1000 --grow_steps 0 0 --fade_in 0 --patch_size 2 \
                --ema_kimg 500 --ema_warmup 0.1 --ema 0.9999 --diff_aug translation,cutout,color \
                --random_seed {args.seed}"

#batch_size = 16



# Exemplo de uso no código
print('\n Starting training', flush=True)
print(f'Total of classes being trained: {len(classes)}\n', flush=True)
print(classes, flush=True)
print('----------------------------------------------------------------------------------------------------', flush=True)

for i, class_name in enumerate([s for s in classes]):
    c = class_name.replace('.csv', '')
    exp_name = f'{c}_{max_iter}_D_{seq_len}_{channels}axis'
    print('\n----------------------------------------------------------------------------------------------------', flush=True)
    print('\n Starting individual training', flush=True)
    print(f'{c} training', flush=True)
    print(f'Data path: {data_path + classes[i]}',flush=True)
    print(f'Label path: {label_path[i]}')
    print('----------------------------------------------------------------------------------------------------', flush=True)

    os.system(f'{default_string}' + f' --class_name {c}' + f' --seq_len {seq_len}'\
          + f' --max_iter {max_iter}' + f' --exp_name {exp_name}' + f' --log_dir {log_dir}'\
          + f' --checkpoint_number {checkpoint_number}' + f' --data_path {data_path + classes[i]}'\
          + f' --channels {channels}' + f' --label_path {label_path[i]}')