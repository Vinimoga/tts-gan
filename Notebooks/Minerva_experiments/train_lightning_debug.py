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
import requests

from torch.utils import data
from torch.utils.data import TensorDataset, DataLoader
from dataLoader import daghar_load_dataset_with_label

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
    parser.add_argument('--clip_grad', type=none_or_float, default=5., help="Valor de clipagem de gradiente ou None")
    parser.add_argument('--save_dir', type=str, default='/workspaces/vinicius.garcia/Projetos/tts-gan/Notebooks/Minerva_experiments/training')
    parser.add_argument('--name', type=str, default='experimentx')
    parser.add_argument('--data_path', type=str, default='/workspaces/vinicius.garcia/Projetos/standardize_view')
    opt = parser.parse_args()

    return opt



class MyDataModule(L.LightningDataModule):
    def __init__(self, train_loader, val_loader, test_loader):
        super().__init__()
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader


args = parse_args()

print(args)

set_seed(args.seed)
device = 'cuda' 

generator = TTSGAN_Generator(seq_len = 60, channels = 6)
discriminator = TTSGAN_Discriminator(seq_len = 60, channels = 6, unsqueeze = True)

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
            toggle = False
            )

train_data_set = daghar_load_dataset_with_label(
    class_name='daghar',
    seq_len=60,
    data_path='/workspaces/vinicius.garcia/Projetos/DAGHAR_split_25_10_all/train/data/UCI_DAGHAR_Multiclass.csv',
    label_path='/workspaces/vinicius.garcia/Projetos/DAGHAR_split_25_10_all/train/label/UCI_Label_Multiclass.csv',
    channels=6,
    percentage=0.8,
)

X = train_data_set[:][0]  # shape: (29430, 6, 1, 60)
y = train_data_set[:][1]  # shape: (29430, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.8,
    random_state=args.seed,
)

# Certifique-se de que estão em formato de Tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).squeeze()
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).squeeze()
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Criando os datasets combinando X e y
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Criando os dataloaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Passando pro seu DataModule
data_module = MyDataModule(train_loader=train_loader, val_loader=test_loader, test_loader=test_loader)

train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

# Exemplo de iteração através do train_loader
print(len(train_loader))
print(len(val_loader))
for batch in train_loader:
    X_batch, y_batch = batch
    print(X_batch.shape, y_batch.shape)  # Deve imprimir torch.Size([64, 6, 60]) torch.Size([64])
    break

printing_callback = MyPrintingCallback()

logger = CSVLogger(save_dir=args.save_dir, name=args.name, version=args.version)

trainer = L.Trainer(accelerator=args.device, devices=1,
                    callbacks=[printing_callback], #[printing_callback, knnvalidation_callback, tsnecallbackencoder],#[printing_callback, knnvalidation_callback, tsnecallbackencoder], #printing_callback
                    logger=logger, max_epochs=args.n_epochs)

trainer.fit(model = model, datamodule = data_module)

torch.save(model.state_dict(), os.path.join(args.save_dir, args.name, f"model_{args.version}.pth"))
print("Training complete. Model saved.")


def send_telegram_message(message):
    token = "8083579885:AAEGyRUnYB86xHtlukUbkzksSHjAsxxDNiU"
    chat_id = "7823372332"  # Substitua com o chat_id que você pegou do JSON
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    response = requests.post(url, data=payload)
    if response.status_code != 200:
        print(f"Erro ao enviar mensagem: {response.text}")
    return response

send_telegram_message("✅ Seu código terminou com sucesso!")
