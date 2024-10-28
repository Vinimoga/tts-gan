# Generator synthetic Running and Jumping data 
# Made them to a Pytorch Dataset 

from torch.utils.data import Dataset, DataLoader
import torch
from GANModels import *
import numpy as np
import os

class Synthetic_Dataset(Dataset):
    def __init__(self, 
                 Jumping_model_path = './pre-trained-models/JumpingGAN_checkpoint',
                 Running_model_path = './pre-trained-models/RunningGAN_checkpoint',
                 sample_size = 1000):
        
        self.sample_size = sample_size
        
        #Generate Running Data
        running_gen_net = Generator(seq_len=150, channels=3, latent_dim=100)
        running_ckp = torch.load(Running_model_path, map_location=torch.device('cpu')) #take off map_location if GPU
        running_gen_net.load_state_dict(running_ckp['gen_state_dict'])
        
        #Generate Jumping Data
        jumping_gen_net = Generator(seq_len=150, channels=3, latent_dim=100)
        jumping_ckp = torch.load(Jumping_model_path,  map_location=torch.device('cpu')) #take off map_location if GPU
        jumping_gen_net.load_state_dict(jumping_ckp['gen_state_dict'])
        
        try:
            print(f'Running checkpoint: {running_ckp["epoch"]}')
            print(f'Jumping checkpoint: {jumping_ckp["epoch"]}')
        except:
            print(f'All epochs are 0')

        #generate synthetic running data label is 0
        z = torch.FloatTensor(np.random.normal(0, 1, (self.sample_size, 100)))
        self.syn_running = running_gen_net(z)
        self.syn_running = self.syn_running.detach().numpy()
        self.running_label = np.zeros(len(self.syn_running))
        
        #generate synthetic jumping data label is 1
        z = torch.FloatTensor(np.random.normal(0, 1, (self.sample_size, 100)))
        self.syn_jumping = jumping_gen_net(z)
        self.syn_jumping = self.syn_jumping.detach().numpy()
        self.jumping_label = np.ones(len(self.syn_jumping))
        
        self.combined_train_data = np.concatenate((self.syn_running, self.syn_jumping), axis=0)
        self.combined_train_label = np.concatenate((self.running_label, self.jumping_label), axis=0)
        self.combined_train_label = self.combined_train_label.reshape(self.combined_train_label.shape[0], 1)
        
        print(self.combined_train_data.shape)
        print(self.combined_train_label.shape)
        
        
    def __len__(self):
        return self.sample_size * 2
    
    def __getitem__(self, idx):
        return self.combined_train_data[idx], self.combined_train_label[idx]
    
class Single_Class_Synthetic_Dataset(Dataset):
    '''
    

    '''
    def __init__(self, 
                 path = './pre-trained-models/RunningGAN_checkpoint',
                 sample_size = 600,
                 seq_len = 150,
                 latent_dim=100,
                 channels = 3,
                 label = 0):
        
        self.sample_size = sample_size
        
        #Generate Data
        gen_net = Generator(seq_len=seq_len, channels=channels, latent_dim=latent_dim)
        ckp = torch.load(path,  map_location=torch.device('cpu')) #take off map_location if GPU
        gen_net.load_state_dict(ckp['gen_state_dict'])
        
        try:
            print(f'Checkpoint epochs: {ckp["epoch"]}')
        except:
            print(f'Epochs: 0')

        #generate synthetic data with label
        z = torch.FloatTensor(np.random.normal(0, 1, (self.sample_size, 100)))
        self.syn = gen_net(z)
        self.syn = self.syn.detach().numpy()
        self.label = np.full(len(self.syn), label)
        
        self.label = self.label.reshape(self.label.shape[0], 1)
        
        print(self.syn.shape)
        print(self.label.shape)
        
        
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        return self.syn[idx], self.label[idx]
    
    
