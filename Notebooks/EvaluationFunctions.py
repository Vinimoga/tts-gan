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

from GANModels import Generator, Discriminator, Encoder

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
                 save_path: str = '../Notebooks/Daghar_TTSGAN_Synthetic_Data/',
                 data_path: str = '../DAGHAR_GANs/',
                 label_path: str = '../DAGHAR_GANs/',
                 show: bool = True,
                 iter: int = 5000,
                 n: int = 10,
                 num_workers: int = 6,
                 channels: int = 3,
                 sample_size=600):
    
        self.data_path = data_path
        self.label_path = label_path
        self.models_path = models_path
        self.class_name = class_name
        self.save_path = save_path
        self.seq_len = seq_len
        self.show = show
        self.iter = iter
        self.channels = channels
        self.num_workers = num_workers
        
        self.save_path = self.save_path + f'{class_name}_{seq_len}_{self.iter}/'
        os.makedirs(self.save_path, exist_ok=True)

        print(f'class name : {self.class_name}')
        print(f'Data path is located in: {self.data_path}')
        print(f'Models path is located in: {self.models_path}')
        print(f'dataset: Daghar')

        print(f' \n Starting evaluation')

        #Extract Original Daghar dataset
        print(f'Original Set:')
        self.original_set = daghar_load_dataset_with_label(class_name=self.class_name, data_path=self.data_path, seq_len=self.seq_len,
                                                            label_path=self.label_path, channels=self.channels)

        if sample_size == 'all':
            self.sample_size = len(self.original_set)
        else:
            self.sample_size = sample_size

        print('\n Synthetic Set:')
        self.syn_set = Single_Class_Synthetic_Dataset(path = self.models_path + 'checkpoint',
                                                      seq_len=self.seq_len, channels=self.channels,
                                                      sample_size = self.sample_size)
        
        cossine_similaritys = []
        #evaluate n times to get variance and average
        for i in range(n):
            #Load DataLoader
            original_data_loader = data.DataLoader(self.original_set, batch_size=1, 
                                                   num_workers=num_workers, shuffle=True)
            syn_data_loader = data.DataLoader(self.syn_set, batch_size=1, 
                                              num_workers=self.num_workers, shuffle=True)

            #Extract random vector from dataloader
            self.syn_data = self.extract_dataloader(syn_data_loader)
            self.original_data, self.original_label = self.extract_dataloader(original_data_loader, l=True)[:len(self.syn_data)]


            #Garant numerical stability and make data into shape (batch, channels, seq_len)
            self.syn_data = torch.from_numpy(self.syn_data).type(torch.float)
            self.original_data = torch.from_numpy(self.original_data).type(torch.float)

            #Get FFT data to evaluate
            self.syn_fft_data = fft.fft(self.syn_data, dim=-1)
            self.original_fft_data = fft.fft(self.original_data, dim=-1)
 
            self.fe_original_data = self.FeatureExtractor(self.original_data.unsqueeze(dim=2))
            self.fe_syn_data = self.FeatureExtractor(self.syn_data.unsqueeze(dim=2))

            self.fe_original_fft_data = self.FeatureExtractor(self.original_data.unsqueeze(dim=2))
            self.fe_syn_fft_data = self.FeatureExtractor(self.syn_fft_data.unsqueeze(dim=2))

            cossine_similaritys.append(self.cossine_similarity(self.fe_original_data, self.fe_syn_data,
                                                                 self.fe_original_fft_data, self.fe_syn_fft_data))
        
        #print(cossine_similaritys, len(cossine_similaritys))
        time_samples = [sample['Time'] for sample in cossine_similaritys]
        frequency_samples = [sample['Frequency'] for sample in cossine_similaritys]

        self.dictionary = {'Time': f"{np.mean(time_samples):.5f} ± {np.std(time_samples):.5f}",
                           'Frequency': f"{np.mean(frequency_samples):.5f} ± {np.std(frequency_samples):.5f}" }

        print(self.dictionary)

        #Color for the three axis in Raw Data Visualization
        c =  [
            "#FF5733",  # Vermelho
            "#337BFF",  # Azul
            "#33FF57",  # Verde
            "#FFD133",  # Amarelo
            "#9B33FF",  # Roxo
            "#33FFF6"   # Ciano
             ]

        
        print(f'saving files in {self.save_path}')
        if self.show:
            self.show_raw_data([self.original_data, self.syn_data], c, 
                               title=f'{self.class_name} Time Domain Raw Data for {self.iter} iter')
            self.show_raw_data([self.original_fft_data, self.syn_fft_data], c, 
                               title=f'{self.class_name} Frequency Domain Raw Data for {self.iter} iter')


        self.tsne_time_data = self.TSNE_visualization(self.original_data, self.syn_data, 
                                title=f'Time domain tSNE: {self.class_name} for {self.iter} iter',
                                show=self.show)
        self.tsne_frequency_data = self.TSNE_visualization(self.original_fft_data, self.syn_fft_data, 
                                title=f'Frequency domain tSNE: {self.class_name} for {self.iter} iter', 
                                show=self.show)   
        
        if sample_size == 'all':
            #print(len(self.original_set))
            self.save_samples(sample_size = len(self.original_set))
        else:
            self.save_samples(sample_size=sample_size)

    def extract_dataloader(self, dataloader, l = False):
        data = []
        label_data = []
        for i, (real_sig, label) in enumerate(dataloader):
            real_sig = real_sig.cpu().detach().numpy()
            sig = real_sig.reshape(real_sig.shape[1], real_sig.shape[3])
            label_data.append(label)
            data.append(sig)
        if l:
            return np.array(data), np.array(label_data)
        return np.array(data)
    
    def show_raw_data(self, array, c, title, r = None, n=5):
        '''
        array should be [real, syn],
        c must be [color 1, color 2, color 3]
        r must be an integer between 0 and (len(real) - n)
        '''
        label = ['real', 'synth']
        activities = ['sit', 'stand', 'walk', 'upstairs', 'downstairs','run']
        temp = array
        if not r:
            #print(f'r can be any number from {0} to {len(temp[0])}')
            r = torch.randint(0,len(temp[0]) - n, size=(1,)).item()
        #print(f'n: {n}, r: {r}')
        fig, axs = plt.subplots(len(temp), n , figsize=(35,5))
        fig.suptitle(title, fontsize=30)
        #fig.suptitle('Running', fontsize=30)
        fig.subplots_adjust(wspace=0.1, hspace=0.4)
        for j in range(len(temp)):
            for k in range(n):
                for l in range(len(temp[0][0])):
                    #print(f'j: {j}, k: {k}, l: {l}')
                    axs[j][k].plot(temp[j][r+k][l][:], c=c[l])
                if j == 0:
                    axs[j][k].set_xlabel(activities[self.original_label[r+k].item()], fontsize=20)
        axs[0][0].set_ylabel(label[0], fontsize=20)
        axs[1][0].set_ylabel(label[1], fontsize=20)

        plt.savefig(self.save_path + title + '.png')

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
        plt.savefig(self.save_path + title + '.png')

        return tsne_results

    def cossine_similarity(self, original, synthetic, original_fft, synthetic_fft): 
        cos_sim = nn.CosineSimilarity(dim=-1)

        # Calcular a similaridade cosseno para cada par de vetores
        time_similarity = cos_sim(original, synthetic) 
        frequency_similarity = cos_sim(original_fft, synthetic_fft)

        # Calcular a média da similaridade cosseno
        average_similarity_original = {'Time': time_similarity.mean().item(),
                                       'Frequency': frequency_similarity.mean().item()} 

        return average_similarity_original
    
    def save_samples(self, sample_size = 600):
        os.makedirs(self.save_path + 'samples', exist_ok=True)
        samples = Single_Class_Synthetic_Dataset(path = self.models_path + 'checkpoint',
                                                 seq_len=self.seq_len, sample_size=sample_size)
              
        samples_dataloader = data.DataLoader(samples, batch_size=1, 
                                              num_workers=self.num_workers, shuffle=True)
        
        samples = self.extract_dataloader(samples_dataloader)
        np.save(self.save_path + 'samples/samples',  arr = samples)


class EncoderEvaluation():
    '''
    
    
    
    
    
    '''
    def __init__(self,
                 models_path: str,
                 class_name: str,
                 seq_len: int,
                 channels: int = 3,
                 save_path = '../Notebooks/Encoder_view/',
                 data_path: str = '../DAGHAR_GANs/',
                 label_path: str = None,
                 show: bool = True):
    
        self.data_path = data_path
        self.label_path = label_path
        self.models_path = models_path
        self.class_name = class_name
        self.channels = channels
        self.save_path = save_path
        self.seq_len = seq_len
        self.show = show
        
        print(f'class name : {self.class_name}')
        print(f'Data path is located in: {self.data_path}')
        print(f'Label path is located in: {self.label_path}')
        print(f'Models path is located in: {self.models_path}')
        print(f'dataset: Daghar')

        print(f' \n Starting Encoder Evaluation')

        #Extract Original Daghar dataset
        print(f'Original Set:')
        self.original_set = daghar_load_dataset_with_label(class_name=self.class_name, seq_len=self.seq_len,
                                                           data_path=self.data_path, label_path = self.label_path,
                                                           channels=self.channels)
        
        #get labels
        labels = self.original_set[:][1]
        #print(labels)

        #get encoder
        self.encoder = self.encoder_separation()
        
        #Transform samples to pass through the encoder
        samples = torch.from_numpy(self.original_set[:][0]).float().to('cuda')
        #print(f'samples shape: {samples.shape}, Labels shape:{labels.shape}')

        forward = self.encoder(samples).cpu().detach().numpy()
        print(f'forward shape: {forward.shape}')

        self.tsne = self.TSNE_visualization(forward, labels, 
                                title=f'tSNE: {self.class_name}',
                                show=self.show, save_path = self.save_path)

    def encoder_separation(self):
        encoder = Encoder(in_channels = self.channels, seq_len = self.seq_len)
        ckp = torch.load(self.models_path, map_location=torch.device("cpu")) 
        keys_to_remove = list(ckp['dis_state_dict'].keys())[-4:] 

        for key in keys_to_remove:
            del ckp['dis_state_dict'][key]
        
        encoder.load_state_dict(ckp['dis_state_dict'])

        return encoder.cuda()

    def TSNE_visualization(self, data, labels, title='t-SNE plot', show=True, save_path=''):
        '''

        Espera-se que os dados originais estejam na forma (batch, channels, timeframe), por exemplo, (2784, 3, 50)
        
        '''
        colors = [
            "#FF5733",  # Vermelho
            "#337BFF",  # Azul
            "#33FF57",  # Verde
            "#FFD133",  # Amarelo
            "#9B33FF",  # Roxo
            "#33FFF6"   # Ciano
        ]
        actions = ['sit', 'stand', 'walk', 'upstairs', 'downstairs', 'run']

        # Garantir que os dados são numpy array e embaralhar com labels
        data = np.asarray(data)
        labels = np.asarray(labels)
        l = len(data)
        idx = np.random.permutation(l)
        data, labels = data[idx], labels[idx]

        # Pré-processamento: média ao longo da dimensão dos canais (dim=1)
        # Para cada batch, reduzimos para uma representação média de forma (2784, 50)
        prep = np.mean(data, axis=1)
        print("Shape após a média por canal:", prep.shape)

        # Análise TSNE
        tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(prep)
        print("Shape do resultado TSNE:", tsne_results.shape)

        # Plotagem
        if not show:
            return tsne_results

        f, ax = plt.subplots(1, figsize=(12,6))

        scatter = plt.scatter(
            tsne_results[:, 0], tsne_results[:, 1], 
            c=[colors[label] for label in labels], alpha=0.6
        )
        # Criar uma legenda customizada para as classes
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10) for i in range(len(actions))]
        ax.legend(handles, actions, title="Actions")

        ax.set_title(title)
        ax.set_xlabel('x-tsne')
        ax.set_ylabel('y-tsne')
        if save_path:
            f.savefig(save_path + title + '.png')

        return tsne_results


    