# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import logging
import operator
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from utils.utils import make_grid, save_image
from tqdm import tqdm
import cv2
import random

# from utils.fid_score import calculate_fid_given_paths
from utils.torch_fid_score import get_fid

# from utils.inception_score import get_inception_scorepython exps/dist1_new_church256.py --node 0022 --rank 0sample

logger = logging.getLogger(__name__)

def train(
    args,
    gen_net: nn.Module,
    dis_net: nn.Module,
    gen_optimizer,
    dis_optimizer,
    gen_avg_param,
    train_loader,
    epoch,
    fixed_z,
    writer_dict=None,
    schedulers=None,
):
    i = 0
    gen_step = 0

    # Train mode
    gen_net.train()
    dis_net.train()

    dis_net.to(args.device)
    gen_net.to(args.device)

    dis_optimizer.zero_grad()
    gen_optimizer.zero_grad()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Move real images to device
        real_imgs = imgs.to(dtype=torch.float32, device=args.device)

        # Sample noise as generator input
        z = torch.tensor(
            np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)),
            dtype=torch.float32,
            device=args.device
        )

        # ---------------------
        #  Train Discriminator
        # ---------------------

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()

        assert (
            fake_imgs.size() == real_imgs.size()
        ), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"
        fake_validity = dis_net(fake_imgs)

        if isinstance(fake_validity, list):
            d_loss = 0
            for real_validity_item, fake_validity_item in zip(real_validity, fake_validity):
                real_label = torch.full_like(real_validity_item, 1.0, device=args.device)
                fake_label = torch.full_like(fake_validity_item, 0.0, device=args.device)

                d_real_loss = nn.MSELoss()(real_validity_item, real_label)
                d_fake_loss = nn.MSELoss()(fake_validity_item, fake_label)
                d_loss += d_real_loss + d_fake_loss
        else:
            real_label = torch.full_like(real_validity, 1.0, device=args.device)
            fake_label = torch.full_like(fake_validity, 0.0, device=args.device)

            d_real_loss = nn.MSELoss()(real_validity, real_label)
            d_fake_loss = nn.MSELoss()(fake_validity, fake_label)
            d_loss = d_real_loss + d_fake_loss

        d_loss.backward()

        if (iter_idx + 1) % args.accumulated_times == 0:
            torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.0)
            dis_optimizer.step() 
            dis_optimizer.zero_grad()
            
           
        # -----------------
        #  Train Generator
        # -----------------

        for accumulated_idx in range(args.g_accumulated_times):
            gen_z = torch.tensor(
                np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)),
                dtype=torch.float32,
                device=args.device
            )

            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            if isinstance(fake_validity, list):
                g_loss = 0
                for fake_validity_item in fake_validity:
                    real_label = torch.full_like(fake_validity_item, 1.0, device=args.device)
                    g_loss += nn.MSELoss()(fake_validity_item, real_label)
            else:
                real_label = torch.full_like(fake_validity, 1.0, device=args.device)
                g_loss = nn.MSELoss()(fake_validity, real_label)
            g_loss.backward()

        torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.0)
        gen_optimizer.step()
        gen_optimizer.zero_grad()
        i += 1
        if i == 368:
            registrar_epocas_txt(epoch + 1, d_loss.item(), g_loss.item())

        # Moving average weight
        ema_nimg = args.ema_kimg * 1000
        cur_nimg = args.batch_size * args.world_size * global_steps
        if args.ema_warmup != 0:
            ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
            ema_beta = 0.5 ** (float(args.batch_size * args.world_size) / max(ema_nimg, 1e-8))
        else:
            ema_beta = args.ema

        # moving average weight
        for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
            cpu_p = deepcopy(p).detach().cpu()
            avg_p.mul_(ema_beta).add_(cpu_p * (1.0 - ema_beta))
            del cpu_p

        # writer.add_scalar('g_loss', g_loss.item(), global_steps) if args.rank == 0 else 0
        gen_step += 1
        
def registrar_epocas_txt(epoca, d_loss, g_loss):
    """
    Registra os valores de var1 e var2 em um arquivo txt no caminho fixo,
    organizados por época.

    Parâmetros:
        epoca (int): número da época atual.
        var1 (float): valor da primeira variável.
        var2 (float): valor da segunda variável.
    """
    caminho_arquivo = '/workspaces/vinicius.garcia/Projetos/tts-gan/log_de_treino/log_pytorch.txt'
    
    with open(caminho_arquivo, 'a') as f:
        f.write(f'Época {epoca}: d_loss = {d_loss:.4f}, g_loss = {g_loss:.4f}\n')

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
        return lr


def load_params(model, new_param, args, mode="gpu"):
    if mode == "cpu":
        for p, new_p in zip(model.parameters(), new_param):
            cpu_p = deepcopy(new_p)
            #             p.data.copy_(cpu_p.cuda().to(f"cuda:{args.gpu}"))
            p.data.copy_(cpu_p.cuda().to("cpu"))
            del cpu_p

    else:
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)


def copy_params(model, mode="cpu"):
    if mode == "gpu":
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def save_state_of_model(model, optimizer, batch, z, loss, final_part_of_path, log_dir='/workspaces/container-workspace/Projetos/tts-gan/debbuging'):
    """
    Save the state of the model to a file.
    :param model: The model to save.
    :param path: The path to save the model state.
    """
    grads = get_gradients(model)
    os.makedirs(log_dir, exist_ok=True) 
    torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(), 
    'gradients': grads,
    'batch': batch,
    'z': z,
    'loss': loss,
    },
    f'{log_dir}/{final_part_of_path}')

def get_gradients(model):
    """
    Get the gradients of the model parameters.
    :param model: The model to get gradients from.
    :return: A dictionary of gradients.
    """
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().cpu().clone()
    return grads