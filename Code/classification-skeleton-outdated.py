import csv
import os 
import sys
import random
import time
import statistics
import psutil

import matplotlib.pyplot as plt

from multiprocessing import Process, Queue

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from DataProcessing.rand_walk_code import *
from train_utils import save_plot, get_props, get_loaders
from models import ResNet, masked_cross_entropy, BasicBlock
from sklearn.metrics import roc_auc_score
from encoding import get_feat_dict


# Hyperparameters
epochs_per_load = 1
num_loads = 1000
feat_nums = [100, 180, 5, 2, 96]
e_sizes = [20, 50, 10, 5, 50]
# filt is a mask for which features you would like to use. Its passed to get_loaders
# to select which features the loader will contain and its passed into resnet because
# the architecture is dependent on the number of features.
# With no filtering at all, at 0.0001 learning rate, gets up to about 0.76 or so val ROC AUC
# and 0.9 train ROC AUC at 100 epochs. This is similar to with previous encoding, and also is
# worse than when a higher learning rate was used.
filt = [True, True, True, True, True]
feat_nums = [feat_nums[a] for a in range(len(feat_nums)) if filt[a]]

# Atom Features
# From "Analyzing Learned Molecular Representations for Property Prediction"
# 1. Atomic Number: atom.GetAtomicNum()     ~100
# 2. Num Neighbors: atom.GetDegree()        4
# 3. Charge:        atom.GetFormalCharge()  5
# 4. Chirality:     atom.GetChiralTag()     3
# 5. NumHs:         atom.GetTotalNumHs()    4 (0-3)
# 6. Hybridization: atom.GetHybridization() 3
# 7. Aromaticity:   atom.GetIsAromatic()    2
# 8. (opt) Mass:    atom.GetMass() * 0.01   float (use truncated float to save csv space)

# Bond Features
# From "Analyzing Learned Molecular Representations for Property Prediction"
# 1. Bond Type      bond.GetBondType        4
# 2. Conjugated     bond.GetIsConjugated()  2
# 3. Ring           bond.IsInRing()         2
# 4. Stereo         bond.GetStereo()        6
# Seen elsewhere
# 5. (opt) Edge Distances                   float

# Encodings   
# 1. Atomic Number                                      ~100 classes     20 features
# 2. Orbital Features (chiral, numhs, degree, hybrid)   144 classes      50 features
# 3. Charge                                             5 classes        10 features
# 4. Aromaticity                                        2 classes        5 features
# 5. Bond Features (bond type, conjugate, ring, stereo) 96 classes       50 features

# -----------------------------------------------------   Train   ------------------------------------------------------------------------- #

def train(working_dir, grid_size, learning_rate, batch_size, num_cores):
	process = psutil.Process(os.getpid())
	print(process.memory_info().rss / 1024 / 1024 / 1024)
	train_feat_dict = get_feat_dict(working_dir + "/train_smiles.csv")
	val_feat_dict = get_feat_dict(working_dir + "/val_smiles.csv")
	test_feat_dict = get_feat_dict(working_dir + "/test_smiles.csv")
	# There are about 0.08 gb
	process = psutil.Process(os.getpid())
	print("pre model")
	print(process.memory_info().rss / 1024 / 1024 / 1024)

	torch.set_default_dtype(torch.float64)
	train_props, val_props, test_props = get_props(working_dir, dtype=int)
	print("pre model post props")
	print(process.memory_info().rss / 1024 / 1024 / 1024)
	model = ResNet(BasicBlock, [2,2,2,2], grid_size, "classification", feat_nums, e_sizes, num_classes=train_props.shape[1])
	model.float()
	model.cuda()
	print("model params")
	pytorch_total_params = sum(p.numel() for p in model.parameters())
	print(pytorch_total_params)
	model.cpu()
	print("model")
	print(process.memory_info().rss / 1024 / 1024 / 1024)
	loss_function = masked_cross_entropy
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	tl_list = []
	vl_list = []
	tmra_list = []
	vmra_list = []

	for file_num in range(num_loads):
		# Get new random walks
		if file_num == 0:
			print("before get_loaders")
			process = psutil.Process(os.getpid())
			print(process.memory_info().rss / 1024 / 1024 / 1024)
			train_loader, val_loader, test_loader = get_loaders(num_cores, \
												working_dir, \
												file_num, \
												grid_size, \
												batch_size, \
												train_props, \
												train_feat_dict, \
												val_props=val_props, \
												val_feat_dict=val_feat_dict, \
												test_props=test_props, \
												test_feat_dict=test_feat_dict)
		else:
			print("before get_loaders 2")
			process = psutil.Process(os.getpid())
			print(process.memory_info().rss / 1024 / 1024 / 1024)
			train_loader, _, _ = get_loaders(num_cores, \
										working_dir, \
										file_num, \
										grid_size, \
										batch_size, \
										train_props, \
										train_feat_dict)
		# Train on a single set of random walks, can do multiple epochs if desired
		for epoch in range(epochs_per_load):
			model.train()
			model.cuda()
			t = time.time()
			train_loss_list = []
			props_list = []
			outputs_list = []
			# change
			for i, (walks_int, walks_float, props) in enumerate(train_loader):
				walks_int = walks_int.cuda()
				walks_int = walks_int.long()
				walks_float = walks_float.cuda()
				walks_float = walks_float.float()
				props = props.cuda()
				props = props.long()
				props_list.append(props)
				outputs = model(walks_int, walks_float)
				outputs_list.append(outputs)
				loss = loss_function(props, outputs)
				train_loss_list.append(loss.item())
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			props = torch.cat(props_list, 0)
			props = props.cpu().numpy()
			outputs = torch.cat(outputs_list, 0)
			outputs = outputs.detach().cpu().numpy()
			# Get train rocauc value
			train_rocaucs = []
			for i in range(props.shape[1]):
				mask = props[:, i] != 2
				train_rocauc = roc_auc_score(props[mask, i], outputs[mask, i])
				train_rocaucs.append(train_rocauc)
			model.eval()
			with torch.no_grad():
				ds = val_loader.dataset 
				walks_int = ds.int_feat_tensor
				walks_float = ds.float_feat_tensor
				props = ds.prop_tensor
				walks_int = walks_int.cuda()
				walks_int = walks_int.long()
				walks_float = walks_float.cuda()
				walks_float = walks_float.float()
				props = props.cuda()
				outputs = model(walks_int, walks_float)
				loss = loss_function(props, outputs)
				props = props.cpu().numpy()
				outputs = outputs.cpu().numpy()
				val_rocaucs = []
				for i in range(props.shape[1]):
					mask = props[:, i] != 2
					val_rocauc = roc_auc_score(props[mask, i], outputs[mask, i])
					val_rocaucs.append(val_rocauc)
			print("load: " + str(file_num) + ", epochs: " + str(epoch))
			print("training loss")
			# Slightly approximate since last batch can be smaller...
			tl = statistics.mean(train_loss_list)
			print(tl)
			print("val loss")
			vl = loss.item()
			print(vl)
			print("train mean roc auc")
			tmra = sum(train_rocaucs) / len(train_rocaucs)
			print(tmra)
			print("val mean roc auc")
			vmra = sum(val_rocaucs) / len(val_rocaucs)
			print(vmra)
			print("time")
			print(time.time() - t)
			tl_list.append(tl)
			vl_list.append(vl)
			tmra_list.append(tmra)
			vmra_list.append(vmra)
			model.cpu()
		file_num += 1
		del train_loader
	save_plot(tl_list, vl_list, tmra_list, vmra_list)
	return model

# -----------------------------------------------------   Main   ------------------------------------------------------------------------- #

# Training script for 
# argv[1]: Directory containing walks
# argv[2]: Grid size
# argv[3]: Learning rate
# argv[4]: Batch size 

print("we are here")
def main():
	print(torch.cuda.is_available())
	working_dir = sys.argv[1]
	grid_size = int(sys.argv[2])
	learning_rate = float(sys.argv[3])
	batch_size = int(sys.argv[4])
	num_cores = int(sys.argv[5])
	print(learning_rate)
	print(batch_size)
	train(working_dir, grid_size, learning_rate, batch_size, num_cores)

if __name__ == "__main__":
	main()

# Ideas:
# Tweak num of embedding layers?

# learning_rates = [0.001, 0.0001, 0.00001]
# batch_sizes = [1, 32, 64, 256]
# grid_size = [16, 32, 64]

# python classification.py 0.0001 64

# learning_rate = 0.001