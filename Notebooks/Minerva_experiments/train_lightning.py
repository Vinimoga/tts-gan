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

from customdataset import CustomDataModule

from lightning.pytorch.callbacks import ModelCheckpoint
from callbacks import TsneGeneratorCallback, TsneEncoderCallback, KNNValidationCallback, MyPrintingCallback

from lightning.pytorch.loggers.csv_logs import CSVLogger
import copy
import random

import argparse

def set_seed(seed: int = 42):  
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    L.seed_everything(seed)

def none_or_float(value):
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float or 'None'")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--n_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--version', type=str, default="1")
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--gen_lr', type=float, default=0.0001)
    parser.add_argument('--dis_lr', type=float, default=0.0003)
    parser.add_argument('--assimetrical_percentage', type=float, default=1.0)
    parser.add_argument('--clip_grad', type=none_or_float, default=None, help="Valor de clipagem de gradiente ou None")
    parser.add_argument('--save_dir', type=str, default='/workspaces/vinicius.garcia/Projetos/tts-gan/Notebooks/Minerva_experiments/training')
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--data_path', type=str, default='/workspaces/vinicius.garcia/Projetos/standardize_view')
    opt = parser.parse_args()

    return opt

args = parse_args()

print(args)

set_seed(args.seed)
device = 'cpu'

SAC = ['Sit', 'Stand', 'Walk', 'Upstairs', 'Downstairs', 'Run']
dataNames = os.listdir(args.data_path)

X = []
y = []   
for dataName in dataNames:
    print(dataName)
    dfTr = pd.read_csv(args.data_path + '/' + dataName + '/train.csv')
    X_tr = dfTr.values[:,:360].reshape(-1,6,60)
    y_tr = dfTr.values[:,-1].astype(np.int32)
    X.append(X_tr)
    y.append(y_tr)
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state = args.seed)

X_train = torch.tensor(X_train.astype(np.float32), dtype=torch.float32, device=args.device).detach()
X_test = torch.tensor(X_test.astype(np.float32), dtype=torch.float32, device=args.device).detach()
y_train = torch.tensor(y_train.astype(np.float32), dtype=torch.float32, device=args.device).detach()
y_test = torch.tensor(y_test.astype(np.float32), dtype=torch.float32, device=args.device).detach()


X = X.astype(np.float32)
y = y.astype(int)
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)

# Crie o DataModule
data_module = CustomDataModule(X_tensor, y_tensor, batch_size=args.batch_size)

# Configure o DataModule
data_module.setup()

# Acesse os dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Exemplo de iteração através do train_loader
print(len(train_loader))
print(len(val_loader))
for batch in train_loader:
    X_batch, y_batch = batch
    print(X_batch.shape, y_batch.shape)  # Deve imprimir torch.Size([64, 6, 60]) torch.Size([64])
    break

#set_seed(args.seed)
generator = TTSGAN_Generator(seq_len = 60, channels = 6)
discriminator = TTSGAN_Discriminator(seq_len = 60, channels = 6)

model = GAN(generator = generator,
            discriminator = discriminator, 
            loss_gen = torch.nn.MSELoss(),
            loss_dis = torch.nn.MSELoss(),
            assimetrical_percentage = args.assimetrical_percentage,
            generator_lr = args.gen_lr,
            discriminator_lr = args.dis_lr,
            beta1 = args.beta1,
            beta2 = args.beta2,
            clip_grad = args.clip_grad,
            )



#checkpoint_callback = ModelCheckpoint()
printing_callback = MyPrintingCallback()
#knnvalidation_callback = KNNValidationCallback(k=10, flatten=True)
#tsnecallbackgenerator = TsneGeneratorCallback(image_save_dir=f'{save_dir}/{name}/version_{version}/tsne/generator/')
#tsnecallbackencoder = TsneEncoderCallback(image_save_dir=f'{save_dir}/{name}/version_{version}/tsne/encoder/', flatten=True)

logger = CSVLogger(save_dir=args.save_dir, name=args.name, version=args.version)

trainer = L.Trainer(accelerator=args.device, devices=1,
                    callbacks=[printing_callback], #[printing_callback, knnvalidation_callback, tsnecallbackencoder],#[printing_callback, knnvalidation_callback, tsnecallbackencoder], #printing_callback
                    logger=logger, max_epochs=args.n_epochs) #max_steps=50000

trainer.fit(model = model, datamodule = data_module)