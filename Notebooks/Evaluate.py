#Important Imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np

import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from LoadRealRunningJumping import *
from LoadSyntheticRunningJumping import *

from dataLoader import *

import torch.fft as fft


class DagharUniclassEvaluation():
    '''
    Original data should be bigger or the same lenght of synthetic data
    
    
    
    
    
    '''

    def __init__(self,
                 models_path,
                 class_name,
                 seq_len,
                 data_path = '/workspaces/container-workspace/DAHAR_GANs/'):
    
        self.data_path = data_path
        self.models_path = models_path
        self.class_name = class_name
        self.seq_len = seq_len

        print(f'Data path is located in: {self.data_path}')
        print(f'Models path is located in: {self.models_path}')
        print(f'dataset: Daghar')

        print(f' \n Starting evaluation')

        #Extract Original Daghar dataset
        self.original_set = daghar_load_dataset(class_name=self.class_name)
        self.syn_set = Single_Class_Synthetic_Dataset(path = self.models_path + 'checkpoint',
                                                      seq_len=self.seq_len)
        
        #Load DataLoader
        original_data_loader = data.DataLoader(self.original_set, batch_size=1, num_workers=1, shuffle=True)
        syn_data_loader = data.DataLoader(self.syn_set, batch_size=1, num_workers=1, shuffle=True)

        #Extract random vector from dataloader
        self.syn_data = self.extract_dataloader(syn_data_loader)
        self.original_data = self.extract_dataloader(original_data_loader)[:len(self.syn_data)]

        #Garant numerical stability and make data into shape (batch, channels, seq_len)
        self.syn_data = torch.from_numpy(self.syn_data).type(torch.float)
        self.original_data = torch.from_numpy(self.original_data).type(torch.float)

        #Get FFT data to evaluate
        self.syn_fft_data = fft.fft(self.syn_data, dim=-1)
        self.original_fft_data = fft.fft(self.original_data, dim=-1)
        
        print('\nwhat is happening')
        
        time_domain = [self.original_data, self.syn_data]
        print(len(time_domain))
        frequency_domain = [self.original_fft_data, self.syn_fft_data]
        c= ['orangered', 'green']
        title = ['Real', 'Synth']

        print('Time Domain Raw Data:')
        self.show_raw_data(time_domain, c, title)

        print('Frequency Domain Raw Data:')
        self.show_raw_data(frequency_domain, c, title)

    
    def extract_dataloader(self, dataloader):
        data = []
        for i, (real_sig, label) in enumerate(dataloader):
            real_sig = real_sig.cpu().detach().numpy()
            sig = real_sig.reshape(real_sig.shape[1], real_sig.shape[3])
            data.append(sig)
    
        return np.array(data)
    
    def show_raw_data(self, array, c, title, r = None):
        if not r:
            r = torch.randint(0,599, size=(1,)).item()
        temp = array
        print(r)
        print(len(temp))
        fig, axs = plt.subplots(1, len(temp), figsize=(35,5))
        #fig.suptitle('Running', fontsize=30)
        fig.subplots_adjust(wspace=0.1, hspace=0.4)
        for j in range(len(temp)):
            print(j)
            for l in range(3):
                axs[j].plot(temp[j][r][l][:], c=c[l])
            axs[j].set_title(title[j], fontsize=20)
