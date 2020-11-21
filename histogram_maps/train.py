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
import pickle

from helper_functions import *
import ROOT


f = uproot.open('/Users/jshlomi/Desktop/Datasets/TruthTagging/dataset5M.root')

n_entries = f['t1'].numentries
training_split = 2000000



jets_train_data = collect_jets(f,0,training_split)
#jet_valid_data = collect_jets(f,training_split,n_entries)


eff_maps = {
    1: None,
    2: None,
    3: None,
}

effmap_errors = {
    1: None,
    2: None,
    3: None,
}


for flav_i, flav in enumerate([1,2,3]):
    flav_cut = np.where(jets_train_data['jet_label']==flav)[0]
    
    pt_values = jets_train_data['jet_pt'].values[flav_cut]
    eta_values = jets_train_data['jet_eta'].values[flav_cut]
    istagged_values = jets_train_data['jet_tag'].values[flav_cut]

    pt_bins = np.linspace(20,600,20)
    eta_bins = np.linspace(-2.5,2.5,11)
    n_pt_bins = len(pt_bins)-1
    n_eta_bins = len(eta_bins)-1

    tagged_jets = np.where(istagged_values ==1)[0]
    total_histogram = np.histogram2d(pt_values,eta_values,bins=(pt_bins,eta_bins))
    pass_histogram = np.histogram2d(pt_values[tagged_jets],eta_values[tagged_jets],bins=(pt_bins,eta_bins))

    hist_pass_th2 = ROOT.TH2F('temp_h_pass','temp_h_pass',n_pt_bins,pt_bins.astype(float),
                                      n_eta_bins,eta_bins.astype(float))
    hist_total_th2 = ROOT.TH2F('temp_h_total','temp_h_total',n_pt_bins,pt_bins.astype(float),
                                      n_eta_bins,eta_bins.astype(float))
    for bin_i in range(n_pt_bins):
        for bin_j in range(n_eta_bins):
            hist_pass_th2.SetBinContent(bin_i+1,bin_j+1, pass_histogram[0][bin_i][bin_j] )
            hist_total_th2.SetBinContent(bin_i+1,bin_j+1, total_histogram[0][bin_i][bin_j] )

    teff_th2 = ROOT.TEfficiency(hist_pass_th2,hist_total_th2)
    #self.teff_th2.SetStatisticOption(ROOT.TEfficiency.kFAC)
    teff_th2.SetStatisticOption(ROOT.TEfficiency.kFWilson)
    teff_th2.SetConfidenceLevel(0.68)

    eff_map = np.divide(pass_histogram[0],total_histogram[0])

    eff_maps[flav] = eff_map.copy()

    effmap_errors[flav] = eff_map.copy()
    for bin_i in range(n_pt_bins):
        for bin_j in range(n_eta_bins):
            global_bin = teff_th2.GetGlobalBin(bin_i+1,bin_j+1)
            effmap_errors[flav][bin_i][bin_j] = np.amax( [teff_th2.GetEfficiencyErrorLow(global_bin), teff_th2.GetEfficiencyErrorUp(global_bin)] )

f = open("effmaps.pkl","wb")
pickle.dump(eff_maps,f)
f.close()

f = open("effmap_errors.pkl","wb")
pickle.dump(effmap_errors,f)
f.close()
