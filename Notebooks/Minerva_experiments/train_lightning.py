from minerva.models.nets.time_series.gans import GAN, TTSGAN_Encoder, TTSGAN_Discriminator, TTSGAN_Generator
from minerva.data.data_module_tools import RandomDataModule
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from lightning.pytorch.callbacks import ModelCheckpoint
from callbacks import TsneGeneratorCallback, TsneEncoderCallback, KNNValidationCallback, MyPrintingCallback

import os
import pandas as pd
import numpy as np

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

experiment_number = 2
version = 1
n_epochs = 300 #300
batch_size = 64 #64
SAC = ['Sit', 'Stand', 'Walk', 'Upstairs', 'Downstairs', 'Run']
beta1 = 0.9
beta2 = 0.999
gen_lr = 0.0001
dis_lr = 0.0003
assimetrical_percentage = 1.0
clip_grad = 5. #None


save_dir = './training'
name = 'without_callback_with_clip_grad_batch_size_16'