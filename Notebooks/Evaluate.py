#Important Imports
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data

import numpy as np
import pandas as pd

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
                 models_path: str,
                 class_name: str,
                 seq_len: int,
                 save_path: str = '../Notebooks/Daghar_TTSGAN_Synthetic_data/',
                 data_path: str = '../DAGHAR_GANs/',
                 show: bool = True,
                 iter: int = 5000):
    
        self.data_path = data_path
        self.models_path = models_path
        self.class_name = class_name
        self.save_path = save_path
        self.seq_len = seq_len
        self.show = show
        self.iter = iter

        print(f'class name : {self.class_name}')
        print(f'Data path is located in: {self.data_path}')
        print(f'Models path is located in: {self.models_path}')
        print(f'dataset: Daghar')

        print(f' \n Starting evaluation')

        #Extract Original Daghar dataset
        print(f'Original Set:')
        self.original_set = daghar_load_dataset(class_name=self.class_name, path=self.data_path)

        print('\n Synthetic Set:')
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
        
        #Adjust labels and titles and datasets visulized
        time_domain = [self.original_data, self.syn_data]
        frequency_domain = [self.original_fft_data, self.syn_fft_data]
        c= ['steelblue', 'orangered', 'green']

        if self.show:
            self.show_raw_data(time_domain, c, title=f'{self.class_name} \
                               Time Domain Raw Data for {self.iter} iter')
            self.show_raw_data(frequency_domain, c, title=f'{self.class_name} \
                               Frequency Domain Raw Data for {self.iter} iter')

        self.TSNE_visualization(self.original_data, self.syn_data, 
                                title=f'Time domain tSNE: {self.class_name} for {self.iter} iter',
                                show=self.show)
        self.TSNE_visualization(self.original_fft_data, self.syn_fft_data, 
                                title=f'Frequency domain tSNE: {self.class_name} for {self.iter} iter', 
                                show=self.show)        
        
        self.fe_original_data = self.FeatureExtractor(self.original_data.unsqueeze(dim=2))
        self.fe_syn_data = self.FeatureExtractor(self.syn_data.unsqueeze(dim=2))

        self.fe_original_fft_data = self.FeatureExtractor(self.original_data.unsqueeze(dim=2))
        self.fe_syn_fft_data = self.FeatureExtractor(self.syn_fft_data.unsqueeze(dim=2))
        #########3for i in range(10):
        self.dictionary = self.cossine_similarity(self.fe_original_data, self.fe_syn_data,
                                                      self.fe_original_fft_data, self.fe_syn_fft_data)
        
        #self.dataframe[f'{self.class_name}'] = pd.DataFrame(self.dictionary)
        #pd.set_option('display.precision', 4)

        print(f'{self.class_name} = {self.dictionary}')

    
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

        data = np.abs(data)

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
        #print(temp.mean(dim=0).shape)

        return temp.mean(dim=0) ##Take mean of a batch 

    def TSNE_visualization(self, original, synthetic, title = 't-SNE plot', random = None, show=True):
        '''

        Data original and synthetic is expected to be in shape (batch, timeframe, channel), for 
        example (600, 30, 3)

        '''
        original = np.transpose(original, (0, 2, 1))
        synthetic = np.transpose(synthetic, (0, 2, 1))

        l = len(original)
        idx = np.random.permutation(l)

        # Data preprocessing
        original = np.asarray(original)
        synthetic = np.asarray(synthetic)  
        #random = np.asarray(random)

        original = np.abs(original[idx])
        synthetic = np.abs(synthetic[idx])
        #random = random[idx]
        

        no, seq_len, dim = original.shape  

        for i in range(l):
            if (i == 0):
                prep = np.reshape(np.mean(original[0,:,:], 1), [1, seq_len])
                prep_hat = np.reshape(np.mean(synthetic[0,:,:], 1), [1, seq_len])
                #prep_random = np.reshape(np.mean(random[0,:,:], 1), [1, seq_len])

            else:
                prep = np.concatenate((prep, 
                                            np.reshape(np.mean(original[i,:,:],1), [1, seq_len])))
                prep_hat = np.concatenate((prep_hat, 
                                            np.reshape(np.mean(synthetic[i,:,:],1), [1, seq_len])))
                #prep_random = np.concatenate((prep_random,
                #                            np.reshape(np.mean(random[i,:,:], 1), [1, seq_len])))
                
        # Do t-SNE Analysis together       
        prep_data_final = np.concatenate((prep, prep_hat), axis = 0) #(prep, prep_hat, prep_random)
        #print(prep_data_final.shape)
        # TSNE anlaysis
        tsne = TSNE(n_components = 2, verbose = 0, perplexity = 40, n_iter = 300)
        tsne_results = tsne.fit_transform(prep_data_final)
        #print(tsne_results.shape)
        # Plotting
        if not show:
            return tsne_results
        
        f, ax = plt.subplots(1)

        plt.scatter(tsne_results[:l,0], tsne_results[:l,1], 
                    c = 'red', alpha = 0.2, label = "Original")
        
        plt.scatter(tsne_results[l:2*l,0], tsne_results[l:2*l,1], 
                    c = 'blue', alpha = 0.2, label = "Synthetic")
        
        #plt.scatter(tsne_results[1200:1800,0], tsne_results[1200:1800,1], 
        #            c = '#ffa600', alpha = 0.2, label = "Random")

        ax.legend()

        plt.title(title)
        plt.xlabel('x-tsne')
        plt.ylabel('y_tsne')
        #         plt.show()    

        plt.show()

    def cossine_similarity(self, original, synthetic, original_fft, synthetic_fft): 
        cos_sim = nn.CosineSimilarity(dim=-1)

        # Calcular a similaridade cosseno para cada par de vetores
        time_similarity = cos_sim(original, synthetic) 
        frequency_similarity = cos_sim(original_fft, synthetic_fft)

        # Calcular a média da similaridade cosseno
        average_similarity_original = {'Time': time_similarity.mean().item(),
                                       'Frequency': frequency_similarity.mean().item()} 

        return average_similarity_original