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



def TrainSingleNetwork(dataloader,path_to_save):
	epochs = 42


	net = JetEfficiencyNet(4,[256,256,256,256],[256,128,50])

	lossfunc = nn.BCELoss()
	optimizer =  optim.Adam(net.parameters(), lr=0.001)

	use_gpu = torch.cuda.is_available()

	if use_gpu:
		net.cuda()
	min_loss = 1000000.0

	for epoch in range(epochs):
		for x,y in tqdm(dataloader):

			if use_gpu:
				x = x.cuda()
				y = y.cuda()

			optimizer.zero_grad()
			output = net(x)
			output = output.view(-1)

			loss = lossfunc(output,y.view(-1)) 

			loss.backward()  
			optimizer.step()

		epoch_loss = 0
		n_batches =0
		for x,y in tqdm(dataloader):
			if use_gpu:
				x = x.cuda()
				y = y.cuda()
			n_batches+=1
			with torch.no_grad():
				output = net(x)
				output = output.view(-1)
				loss = lossfunc(output,y.view(-1)) 
				
				epoch_loss+=loss.item()
		epoch_loss = epoch_loss/n_batches

		if epoch_loss < min_loss:
			min_loss = epoch_loss
			torch.save(net.state_dict(), path_to_save)

	return 




