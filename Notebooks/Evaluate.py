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

from sklearn.manifold import TSNE


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
        print(f'Original Set:')
        self.original_set = daghar_load_dataset(class_name=self.class_name)

        print('\n Synthetic Set:')
        self.syn_set = Single_Class_Synthetic_Dataset(path = self.models_path + 'checkpoint',
                                                      seq_len=self.seq_len)
        
        #Load DataLoader
        original_data_loader = data.DataLoader(self.original_set, batch_size=1, num_workers=1, shuffle=True)
        syn_data_loader = data.DataLoader(self.syn_set, batch_size=1, num_workers=1, shuffle=True)

        #Extract random vector from dataloader
        self.syn_data = self.extract_dataloader(syn_data_loader)
        self.original_data = self.extract_dataloader(original_data_loader)[:len(self.syn_data)]
        print(self.syn_data.shape)

        #Garant numerical stability and make data into shape (batch, channels, seq_len)
        self.syn_data = torch.from_numpy(self.syn_data).type(torch.float)
        self.original_data = torch.from_numpy(self.original_data).type(torch.float)
        print(self.syn_data.shape)

        #Get FFT data to evaluate
        self.syn_fft_data = fft.fft(self.syn_data, dim=-1)
        self.original_fft_data = fft.fft(self.original_data, dim=-1)
        
        #Adjust labels and titles and datasets visulized
        time_domain = [self.original_data, self.syn_data]
        frequency_domain = [self.original_fft_data, self.syn_fft_data]
        c= ['steelblue', 'orangered', 'green']

        self.show_raw_data(time_domain, c, title='Time Domain Raw Data')
        self.show_raw_data(frequency_domain, c, title='Frequency Domain Raw Data')

    
    def extract_dataloader(self, dataloader):
        data = []
        for i, (real_sig, label) in enumerate(dataloader):
            real_sig = real_sig.cpu().detach().numpy()
            sig = real_sig.reshape(real_sig.shape[1], real_sig.shape[3])
            data.append(sig)
    
        return np.array(data)
    
    def show_raw_data(self, array, c, title, r = None, n=5):
        '''
        array should be [real, syn],
        c must be [color 1, color 2, color 3]
        r must be an integer between 0 and (len(real) - n)
        '''
        label = ['real', 'synth']
        temp = array
        if not r:
            r = torch.randint(0,len(temp[0]) - n, size=(1,)).item()

        fig, axs = plt.subplots(len(temp), n , figsize=(35,5))
        fig.suptitle(title, fontsize=30)
        #fig.suptitle('Running', fontsize=30)
        fig.subplots_adjust(wspace=0.1, hspace=0.4)
        for j in range(len(temp)):
            for k in range(n):
                for l in range(3):
                    axs[j][k].plot(temp[j][r+k][l][:], c=c[l])
        axs[0][0].set_ylabel(label[0], fontsize=20)
        axs[1][0].set_ylabel(label[1], fontsize=20)

    def rootMeanSquare(self, data: torch.tensor):
        return torch.sqrt((data**2).mean(dim=-1))
    
    def FeatureExtractor(self, data: torch.tensor, dim: int = -1):
        '''
        A simple feature extractor of several meaningful features from each input
        data sequence. They are the median, mean, standard deviation, variance, root
        mean square, maximum, and minimum values of each input sequence. 

        The tensor is expected to be in shape (600, 3, 1, 150), with 600 of batch, 
        3 channels (x,y,z), height 1 and width 150 (time)

        parameters:
            input:
                data -> The tensor from which we will extract the features
                dim -> dimention of the time channels (for our purposes is the last one)
        '''
        #print(data.shape)

        fmedian = data.median(dim=dim)[0]
        #print(f'median shape: {fmedian.shape}')
        fmean = data.mean(dim=dim)
        #print(f'mean shape: {fmean.shape}')
        fstd = data.std(dim=dim)
        #print(f'STD shape: {fstd.shape}')
        fvar = data.var(dim=dim)
        #print(f'Var shape: {fvar.shape}')
        frms = self.rootMeanSquare(data)
        #print(f'RMS shape: {frms.shape}')
        fmax = data.max(dim=dim)[0]
        #print(f'max value shape: {fmax.shape}')
        fmin = data.min(dim=dim)[0]
        #print(f'min value shape: {fmin.shape}')
        
        temp = torch.cat((fmedian, fmean, fstd, fvar, frms, fmax, fmin), dim=-1)
        #print(temp.shape)

        return temp.mean(dim=0) ##Take mean of a batch 