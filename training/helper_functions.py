import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

import math
import itertools

def collect_jets(f,entrystart,entry_stop):
	df = f['t1'].pandas.df(['jet_pt','jet_eta','jet_phi','jet_label','jet_eff','jet_score','jet_tag'],
						   entrystart=entrystart,entrystop=entry_stop,flatten=False).reset_index()

	df['njets'] = [len(x) for x in df.jet_pt.values ]

	return df
