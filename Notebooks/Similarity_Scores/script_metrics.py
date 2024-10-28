import numpy as np
from metrics.discriminative_score_metrics import discriminative_score_metrics
from metrics.predictive_score_metrics import predictive_score_metrics
import sys
import os

# Adiciona o diretório 'tts-gan' ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from LoadSyntheticRunningJumping import Single_Class_Synthetic_Dataset
from dataLoader import daghar_load_dataset

from torch.utils import data

from tqdm import tqdm

import sys
import json

def main():
    #-----------------------------------------------------------------------------------------
    real_sample = daghar_load_dataset(class_name='run', path='../../DAGHAR_GANs/')
    synthetic_sample= Single_Class_Synthetic_Dataset( path = '../../pre-trained-models/daghar_50000_30_100/run_50000_D_30_2024_10_18_01_39_30/Model/checkpoint',
                                                      sample_size = len(real_sample),
                                                      seq_len = 30)

    real_sample, synthetic_sample = np.array(real_sample[:][0]).squeeze(), np.array(synthetic_sample[:][0]).squeeze()

    batch_size, _, _ = np.asarray(real_sample).shape
    metric_results = dict()
    print('\n\n\n well, at least it is trying')
    metric_results['discriminative']  = discriminative_score_metrics(real_sample, synthetic_sample, batch_size)
    print('\n\n\n well, at least it is trying')
    metric_results['predictive'] = predictive_score_metrics(real_sample, synthetic_sample,batch_size)
    print('\n\n\n well, at least it is trying')
    print("Discriminative: %f +/- %f"%(metric_results['discriminative'][0],metric_results['discriminative'][1]))
    print("Predictive: %f +/- %f"%(metric_results['predictive'][0],metric_results['predictive'][1]))

if __name__ == '__main__':
    main()


    
    


    

#Full credits goes to [Fabiana Clemented](https://towardsdatascience.com/synthetic-time-series-data-a-gan-approach-869a984f2239) for this implementation.<br>
#Paper on [TimeGAN](https://papers.nips.cc/paper/2019/file/c9efe5f26cd17ba6216bbe2a7d26d490-Paper.pdf)
    
