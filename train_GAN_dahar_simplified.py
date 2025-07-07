from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg

# import models_search
# import datasets
from dataLoader import *

# from GANModels import *
# from GANModelsOriginal import *

from torch.utils.data import TensorDataset, DataLoader
from minerva.models.nets.time_series.gans import TTSGAN_Generator as Generator
from minerva.models.nets.time_series.gans import TTSGAN_Discriminator as Discriminator

from functions_simplified import (
    train,
    load_params,
    copy_params,
)
from utils.utils import set_log_dir, save_checkpoint, create_logger

import csv
import os

# from utils.inception_score import _init_inception
# from utils.fid_score import create_inception_graph, check_or_download_inception

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data.distributed
from torch.utils import data
import os
import numpy as np
import torch.nn as nn

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from copy import deepcopy
from adamw import AdamW
import random
import matplotlib.pyplot as plt
import io
import PIL.Image
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split

from customdataset import CustomDataModule, get_data
from argparse import Namespace

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
import lightning as L


def main():
    # args = cfg.parse_args()
    # dict_args = vars(args)
    # print(dict_args)

    args = {
        "accumulated_times": 1,
        "arch": None,
        "augment_times": None,
        "baseline_decay": 0.9,
        "batch_size": 64,
        "beta1": 0.9,
        "beta2": 0.999,
        "bottom_width": 8,
        "channels": 6,
        "checkpoint_number": 4,
        "class_name": "UCI_DAGHAR_Multiclass",
        "controller": "controller",
        "ctrl_lr": 0.00035,
        "ctrl_sample_batch": 1,
        "ctrl_step": 30,
        "d_act": "gelu",
        "d_depth": 3,
        "D_downsample": "avg",
        "d_heads": 4,
        "d_lr": 0.0003,
        "d_mlp": 4,
        "d_norm": "ln",
        "d_spectral_norm": False,
        "d_window_size": 8,
        "data_path": '/workspaces/vinicius.garcia/Projetos/DAGHAR_split_25_10_all/train/data/UCI_DAGHAR_Multiclass.csv', #"/workspaces/vinicius.garcia/Projetos/DAGHAR_split_25_10_all/train/data/UCI_DAGHAR_Multiclass.csv",
        "dataset": "daghar",
        "df_dim": 384,
        "diff_aug": "translation,cutout,color",
        "dis_bs": 64,
        "dis_model": "my_dis",
        "dist_backend": "nccl",
        "dist_url": "tcp://localhost:4321",
        "dropout": 0.0,
        "dynamic_reset_threshold": 0.001,
        "dynamic_reset_window": 500,
        "device": "cuda", #cpu
        "ema_kimg": 500,
        "ema_warmup": 0.1,
        "ema": 0.9999,
        "entropy_coeff": 0.001,
        "eval_batch_size": 100,
        "exp_name": "UCI_DAGHAR_Multiclass_137694_D_60_6axis",
        "fade_in": 0.0,
        "fid_stat": "None",
        "g_accumulated_times": 1,
        "g_act": "gelu",
        "g_depth": "5,4,2",
        "g_lr": 0.0001,
        "g_mlp": 4,
        "g_norm": "ln",
        "g_spectral_norm": False,
        "g_window_size": 8,
        "gen_bs": 64,
        "gen_model": "my_gen",
        "gf_dim": 1024,
        "gpu": None,
        "grow_step1": 25,
        "grow_step2": 55,
        "grow_steps": [0, 0],
        "hid_size": 100,
        "img_size": 32,
        "init_type": "xavier_uniform",
        "label_path": '/workspaces/vinicius.garcia/Projetos/DAGHAR_split_25_10_all/train/label/UCI_Label_Multiclass.csv', #"/workspaces/vinicius.garcia/Projetos/DAGHAR_split_25_10_all/train/label/UCI_Label_Multiclass.csv",
        "latent_dim": 100,
        "latent_norm": False,
        "load_path": None,
        "loca_rank": -1,
        "log_dir": '/workspaces/vinicius.garcia/Projetos/tts-gan/logs/TEST0',#"workspaces/vinicius.garcia/Projetos/tts-gan/logs/TEST",
        "loss": "lsgan",
        "lr_decay": False,
        "max_epoch": 300,
        "max_iter": 137694,
        "max_search_iter": 90,
        "ministd": False,
        "multiprocessing_distributed": False,
        "n_classes": 0,
        "n_critic": 1,
        "num_candidate": 10,
        "num_eval_imgs": 50000,
        "num_landmarks": 64,
        "num_workers": 6,
        "optimizer": "adam",
        "patch_size": 2,
        "path_helper": None,
        "phi": 1.0,
        "print_freq": 1000,
        "random_seed": 3,
        "rank": 0,
        "rl_num_eval_img": 5000,
        "seed": 3,
        "seq_len": 60,
        "shared_epoch": 15,
        "show": False,
        "topk": 5,
        "val_freq": 20,
        "wd": 0.001,
        "world_size": 1,
    }
    args = Namespace(**args)
    
    print(args)

    if args.seed is not None:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        L.seed_everything(args.seed, workers=True)

    if args.gpu is not None:
        # Warning.warn('You have chosen a specific GPU. This will completely disable data parallelism.')
        pass

    main_worker(args.gpu, 1, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # import network
    gen_net = Generator(
        seq_len=args.seq_len, channels=args.channels
    )  # Generator_original
    print(gen_net)
    dis_net = Discriminator(
        seq_len=args.seq_len, channels=args.channels, unsqueeze=False
    )  # Discriminator_original
    print(dis_net)
    if not torch.cuda.is_available():  # just to see the cpu in action
        print("using CPU, this will be slow")

    gen_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gen_net.parameters()),
        args.g_lr,
        (args.beta1, args.beta2),
    )
    dis_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, dis_net.parameters()),
        args.d_lr,
        (args.beta1, args.beta2),
    )
    print(args.class_name)
    print(args.dataset)
    train_data_set = daghar_load_dataset_with_label(
        class_name=args.class_name,
        seq_len=args.seq_len,
        data_path=args.data_path,
        label_path=args.label_path,
        channels=args.channels,
        percentage=0.8,
    )  # Change the percentage if you want
    X = train_data_set[:][0]  # shape: (29430, 6, 1, 60)
    y = train_data_set[:][1]  # shape: (29430, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        train_size=0.8,
        random_state=args.seed,
    )

    # Certifique-se de que estão em formato de Tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Criando os datasets combinando X e y
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Criando os dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(len(train_loader))
    for batch in train_loader:
        X_batch, y_batch = batch
        print(X_batch.shape, y_batch.shape)  # Deve imprimir torch.Size([64, 6, 60]) torch.Size([64])
        break

    t_checkpoint = int(np.ceil(args.max_epoch / args.checkpoint_number))
    fixed_z = 0
    avg_gen_net = deepcopy(gen_net).cpu()
    gen_avg_param = copy_params(avg_gen_net)
    del avg_gen_net
    start_epoch = 0
    best_fid = 1e4

    writer_dict = {
        "train_global_steps": start_epoch * len(train_loader),
    }
    args.path_helper = set_log_dir(args.log_dir, args.exp_name)
    # train loop
    print(f"Epochs between checkpoint: {t_checkpoint}")
    t = 0
    for epoch in range(int(start_epoch), int(args.max_epoch)):
        
        train(
            args=args,
            gen_net=gen_net,
            dis_net=dis_net,
            gen_optimizer=gen_optimizer,
            dis_optimizer=dis_optimizer,
            gen_avg_param=gen_avg_param,
            train_loader=train_loader,
            epoch=epoch,
            fixed_z=fixed_z,
            writer_dict=writer_dict,
            schedulers=None,
        )

        is_best = False
        #avg_gen_net = deepcopy(gen_net)
        #load_params(avg_gen_net, gen_avg_param, args)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "gen_model": args.gen_model,
                "dis_model": args.dis_model,
                "gen_state_dict": gen_net.state_dict(), #gen_net.module.state_dict(),
                "dis_state_dict": dis_net.state_dict(), #dis_net.module.state_dict(),
                #"avg_gen_state_dict": avg_gen_net.state_dict(),#avg_gen_net.module.state_dict()
                "gen_optimizer": gen_optimizer.state_dict(),
                "dis_optimizer": dis_optimizer.state_dict(),
                "best_fid": best_fid,
                "path_helper": args.path_helper,
                "fixed_z": fixed_z,
            },
            is_best,
            args.path_helper["ckpt_path"],
            filename="checkpoint",
        )
        #del avg_gen_net


def gen_plot(gen_net, epoch, class_name):
    """Create a pyplot plot and save to buffer."""
    synthetic_data = []

    for i in range(10):
        fake_noise = torch.FloatTensor(np.random.normal(0, 1, (1, 100)))
        fake_sigs = gen_net(fake_noise).to("cpu").detach().numpy()
        synthetic_data.append(fake_sigs)

    fig, axs = plt.subplots(2, 5, figsize=(20, 5))
    fig.suptitle(f"Synthetic {class_name} at epoch {epoch}", fontsize=30)
    for i in range(2):
        for j in range(5):
            axs[i, j].plot(synthetic_data[i * 5 + j][0][0][0][:])
            axs[i, j].plot(synthetic_data[i * 5 + j][0][1][0][:])
            axs[i, j].plot(synthetic_data[i * 5 + j][0][2][0][:])
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    # added because creates lots of data and broke the node containing the training (isn't made for big training)
    plt.close()  # originally without this
    return buf

if __name__ == "__main__":
    main()
