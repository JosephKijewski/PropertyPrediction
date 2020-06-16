import os
import torch
import time
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch import from_numpy

from multiprocessing import shared_memory, Process, Lock
from multiprocessing import cpu_count, current_process

import psutil
import tracemalloc

# -----------------------------------------------------   Dataset   ------------------------------------------------------------------------- #

def chunk(tot_size, cores):
	chunk_size = float(tot_size) / cores
	lows = [int(chunk_size * i) for i in range(cores)]
	highs = [int(chunk_size * i) for i in range(1, cores + 1)]
	return lows, highs

# Dataset for random walk -> property neural network.

class WalkDataset(Dataset):

	def __init__(self, int_feat_tensor, float_feat_tensor, prop_tensor):
		self.int_feat_tensor = int_feat_tensor
		self.float_feat_tensor = float_feat_tensor 
		self.prop_tensor = prop_tensor
		self.len = self.int_feat_tensor.shape[0]

	def __len__(self):
		return self.len

	def __getitem__(self, index):
		return self.int_feat_tensor[index, :, :, :], self.float_feat_tensor[index, :, :, :], self.prop_tensor[index, :]

# -----------------------------------------------------   Loading Functions   ------------------------------------------------------------------------- #

# Function that loads in properties from working directory.

def get_props(working_dir, dtype):
	train_csv_fn = working_dir + "/train_props.csv"
	val_csv_fn = working_dir + "/val_props.csv"
	test_csv_fn = working_dir + "/test_props.csv"
	train_csv = from_numpy(np.loadtxt(train_csv_fn, dtype=dtype, delimiter=','))
	val_csv = from_numpy(np.loadtxt(val_csv_fn, dtype=dtype, delimiter=','))
	test_csv = from_numpy(np.loadtxt(test_csv_fn, dtype=dtype, delimiter=','))
	return train_csv, val_csv, test_csv

# Create dataloaders given props and working directory.

def get_loaders(working_dir, file_num, grid_size, batch_size, train_props, val_props=None, test_props=None):
	train_file_int = open(working_dir + "/train_walk_files" + str(grid_size) + "/" + str(file_num) + "i.pkl", "rb")
	train_file_float = open(working_dir + "/train_walk_files" + str(grid_size) + "/" + str(file_num) + "f.pkl", "rb")
	train_int_feat_tensor = pickle.load(train_file_int)
	train_float_feat_tensor = pickle.load(train_file_float)
	train_file_int.close()
	train_file_float.close()
	train_ds = WalkDataset(train_int_feat_tensor, train_float_feat_tensor, train_props)
	if type(val_props) != type(None):
		val_file_int = open(working_dir + "/val_walk_files" + str(grid_size) + "/" + str(file_num) + "i.pkl", "rb")
		val_file_float = open(working_dir + "/val_walk_files" + str(grid_size) + "/" + str(file_num) + "f.pkl", "rb")
		val_int_feat_tensor = pickle.load(val_file_int)
		val_float_feat_tensor = pickle.load(val_file_float)
		val_file_int.close()
		val_file_float.close()
		val_ds = WalkDataset(val_int_feat_tensor, val_float_feat_tensor, val_props)
	if type(test_props) != type(None):
		test_file_int = open(working_dir + "/test_walk_files" + str(grid_size) + "/" + str(file_num) + "i.pkl", "rb")
		test_file_float = open(working_dir + "/test_walk_files" + str(grid_size) + "/" + str(file_num) + "f.pkl", "rb")
		test_int_feat_tensor = pickle.load(test_file_int)
		test_float_feat_tensor = pickle.load(test_file_float)
		test_file_int.close()
		test_file_float.close()
		test_ds = WalkDataset(test_int_feat_tensor, test_float_feat_tensor, test_props)
	train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
	val_dl = None
	test_dl = None
	if type(val_props) != type(None):
		val_dl = DataLoader(val_ds, batch_size=batch_size)
	if type(test_props) != type(None):
		test_dl = DataLoader(test_ds, batch_size=batch_size)
	return train_dl, val_dl, test_dl

# Save plot with training loss, validation loss, over time, and training mean roc auc, val mean roc auc if applicable.

def save_plot(tl_list, vl_list, tmra_list=None, vmra_list=None):
	tl_arr = np.array(tl_list)
	vl_arr = np.array(vl_list)
	tmra_arr = np.array(tmra_list)
	vmra_arr = np.array(vmra_list)
	fig, axes = plt.subplots(nrows=2, ncols=1)
	axes[0].scatter(x, tl_arr, color='red', label='train loss')
	axes[0].scatter(x, vl_arr, color='green', label = 'val loss')
	if tmra_list != None:
		axes[1].scatter(x, tmra_arr, color='blue', label= 'train mean rocauc')
	if vmra_list != None:
		axes[1].scatter(x, vmra_arr, color='yellow', label= 'val mean rocauc')
	axes[0].legend(loc=2)
	axes[0].set_ylim(0.1, 0.27)
	if tmra_list != None or vmra_list != None:
		axes[1].legend(loc=2)
		axes[1].set_ylim(0.65, 0.95)
	plt.savefig('plot.png')