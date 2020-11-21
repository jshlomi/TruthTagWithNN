import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import itertools
import math


class JetEventsDataset(Dataset):
    def __init__(self, df,var_transformations):
        self.df = df
        
   
        self.n_events = len(self.df)
        
        
        self.event_njets = df['njets'].values
        
        for col in ['jet_pt','jet_eta','jet_phi']:#,'jet_label']:
            mean, std = var_transformations[col]['mean'],  var_transformations[col]['std']
            flat_array = np.concatenate( self.df[col] )
            flat_array  = (flat_array-mean)/std
            split_array = np.split( flat_array , np.cumsum(self.event_njets)[:-1] )
            self.df[col] = split_array
    

    def __len__(self):
       
        return self.n_events


    def __getitem__(self, idx):
        x = torch.FloatTensor( np.column_stack( (self.df.iloc[idx].jet_pt, self.df.iloc[idx].jet_eta,self.df.iloc[idx].jet_phi,self.df.iloc[idx].jet_label ) ) )
        y = torch.FloatTensor( self.df.iloc[idx].jet_tag )
        
        return x, y


class JetEventsSampler(Sampler):
    def __init__(self, n_nodes_array, batch_size):
        """
        Initialization
        :param n_nodes_array: array of sizes of the events
        """
        super().__init__(n_nodes_array.size)

        self.dataset_size = n_nodes_array.size
        self.batch_size = batch_size

        self.index_to_batch = {}
        self.node_size_idx = {}
        running_idx = -1

        for n_nodes_i in set(n_nodes_array):

            self.node_size_idx[n_nodes_i] = np.where(n_nodes_array == n_nodes_i)[0]

            n_of_size = len(self.node_size_idx[n_nodes_i])
            n_batches = max(n_of_size / self.batch_size, 1)

            self.node_size_idx[n_nodes_i] = np.array_split(np.random.permutation(self.node_size_idx[n_nodes_i]),
                                                           n_batches)
            for batch in self.node_size_idx[n_nodes_i]:
                running_idx += 1
                self.index_to_batch[running_idx] = batch

        self.n_batches = running_idx + 1

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        batch_order = np.random.permutation(np.arange(self.n_batches))
        for i in batch_order:
            yield self.index_to_batch[i]