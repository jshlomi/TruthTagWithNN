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
import json

from helper_functions import *
from model import *
from data_loader import *
from train_single_network import *


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


f = uproot.open('../data/dataset5M.root')

n_entries = f['t1'].numentries
sample_size = 100000


jets_train_data = collect_jets(f,0,sample_size)


n_bootstraps = 100
n_networks_per_bootstrap = 25
batch_size = 5000

if not os.path.exists('networks'):
	os.mkdir('networks')

for bootstrap_i in range(n_bootstraps):

	bootstrap = np.random.choice(sample_size,size=sample_size,replace=True)

	if not os.path.exists('experiments'):
		os.mkdir('experiments')

	if not os.path.exists('experiments/bootstrap_'+str(bootstrap_i)):
		os.mkdir('experiments/bootstrap_'+str(bootstrap_i))


	bootstrapped_data = jets_train_data.iloc[bootstrap].copy().reset_index()

	var_transformations = {}
	for var_i,var  in enumerate(['jet_pt','jet_eta','jet_phi']):
	    var_values = np.concatenate( bootstrapped_data[var].values )
	    var_transformations[var] = {'mean' : float(np.mean(var_values)), 'std' : float(np.std(var_values))}

	if not os.path.exists('var_transforms'):
		os.mkdir('var_transforms')
	with open('var_transforms/var_transformations_bootstrap_'+str(bootstrap_i)+'.json', 'w') as fp:
		json.dump(var_transformations, fp,indent=4)

	ds = JetEventsDataset(bootstrapped_data,var_transformations)
	batch_sampler = JetEventsSampler(ds.event_njets,batch_size)
	data_loader= DataLoader(ds,batch_sampler=batch_sampler,num_workers=0)

	if not os.path.exists('networks/bootstrap_'+str(bootstrap_i)):
		os.mkdir('networks/bootstrap_'+str(bootstrap_i))

	for n_i in range(n_networks_per_bootstrap):

		path_to_save = 'networks/bootstrap_'+str(bootstrap_i)+'/net_'+str(n_i)+'.pt'

		TrainSingleNetwork(data_loader,path_to_save)
