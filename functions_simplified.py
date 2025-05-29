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


def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(
        real_samples.get_device()
    )
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(
        real_samples.get_device()
    )
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty

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
    # writer = writer_dict['writer']
    i = 0
    gen_step = 0
    # train mode
    gen_net.train()
    dis_net.train()

    dis_optimizer.zero_grad()
    gen_optimizer.zero_grad()
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

       # Ensure imgs is a tensor, and convert it explicitly to float32 on CPU
        real_imgs = imgs.to(dtype=torch.float32, device='cpu')

        # Sample noise as generator input, float32 on CPU
        z = torch.tensor(
            np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)),
            dtype=torch.float32,
            device='cpu'
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
            for real_validity_item, fake_validity_item in zip(
                real_validity, fake_validity
            ):
                real_label = torch.full(
                    (real_validity_item.shape[0], real_validity_item.shape[1]),
                    1.0,
                    dtype=torch.float,
                    device=real_imgs.get_device(),
                )
                fake_label = torch.full(
                    (real_validity_item.shape[0], real_validity_item.shape[1]),
                    0.0,
                    dtype=torch.float,
                    device=real_imgs.get_device(),
                )
                d_real_loss = nn.MSELoss()(real_validity_item, real_label)
                d_fake_loss = nn.MSELoss()(fake_validity_item, fake_label)
                d_loss += d_real_loss + d_fake_loss
        else:
            real_label = torch.full(
                (real_validity.shape[0], real_validity.shape[1]),
                1.0,
                dtype=torch.float,
            )  
            fake_label = torch.full(
                (real_validity.shape[0], real_validity.shape[1]),
                0.0,
                dtype=torch.float,
            )  
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
                device='cpu'
            )

            gen_z = gen_z.to(args.device)
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)
            if isinstance(fake_validity, list):
                g_loss = 0
                for fake_validity_item in fake_validity:
                    real_label = torch.full(
                        (
                            fake_validity_item.shape[0],
                            fake_validity_item.shape[1],
                        ),
                        1.0,
                        dtype=torch.float,
                        device="cpu",
                    )
                    g_loss += nn.MSELoss()(fake_validity_item, real_label)
            else:
                real_label = torch.full(
                    (fake_validity.shape[0], fake_validity.shape[1]),
                    1.0,
                    dtype=torch.float,
                    device="cpu",
                )
                g_loss = nn.MSELoss()(fake_validity, real_label)
            g_loss.backward()

        torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.0)
        gen_optimizer.step()
        gen_optimizer.zero_grad()
        i = i+1
        if i == 368:
            registrar_epocas_txt(epoch+1, d_loss, g_loss)
        
        
        # moving average weight
        ema_nimg = args.ema_kimg * 1000
        cur_nimg = args.batch_size * args.world_size * global_steps
        if args.ema_warmup != 0:
            ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
            ema_beta = 0.5 ** (
                float(args.batch_size * args.world_size) / max(ema_nimg, 1e-8)
            )
        else:
            ema_beta = args.ema

        # moving average weight
        for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
            cpu_p = deepcopy(p)
            avg_p.mul_(ema_beta).add_(1.0 - ema_beta, cpu_p.cpu().data)
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
    caminho_arquivo = '/workspaces/container-workspace/tts-gan/log_de_treino/log_pytorch.txt'
    
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