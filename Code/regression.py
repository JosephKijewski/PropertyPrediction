import csv
import os 
import sys
import random
import time
import statistics
import psutil
import pickle

import matplotlib.pyplot as plt
import rdkit.Chem as Chem

from multiprocessing import Process, Queue

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from DataProcessing.rand_walk_code import *
from train_utils import save_plot, get_props, get_loaders
from models import ResNet, masked_cross_entropy, BasicBlock, Bottleneck, densenet201, densenet121, densenet169, densenet161

# How loading is going to work
# There is going to be some sort of apply along axis function that looks up the atom and bond features
# in a dictionary.
# There is going to be a stage where the dictionary is calculated. It should be precalculated at the beginning
# and fed to each dataset. Then, get_item for index i should look at the ith entry of the dictinoary then
# do the apply along axis


# Hyperparameters
epochs_per_load = 1
num_loads = 10000
feat_nums = [100, 180, 5, 2, 96]
# e_sizes = [20, 50, 10, 5, 50]
e_sizes = [40, 900, 25, 10, 480]
# filt is a mask for which features you would like to use. Its passed to get_loaders
# to select which features the loader will contain and its passed into resnet because
# the architecture is dependent on the number of features.
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

def train(working_dir, grid_size, learning_rate, batch_size, num_walks, model_type, fn):
	train_props, val_props, test_props = get_props(working_dir, dtype=np.float32)
	means_stds = np.loadtxt(working_dir + "/means_stds.csv", dtype=np.float32, delimiter=',')

	# filter out redundant qm8 properties
	if train_props.shape[1] == 16:
		filtered_labels = list(range(0, 8)) + list(range(12, 16))
		train_props = train_props[:, filtered_labels]
		val_props = val_props[:, filtered_labels]
		test_props = test_props[:, filtered_labels]

		means_stds = means_stds[:, filtered_labels]
	if model_type == "resnet18":
		model = ResNet(BasicBlock, [2, 2, 2, 2], grid_size, "regression", feat_nums, e_sizes, num_classes=train_props.shape[1])
	elif model_type == "resnet34":
		model = ResNet(BasicBlock, [3, 4, 6, 3], grid_size, "regression", feat_nums, e_sizes, num_classes=train_props.shape[1])
	elif model_type == "resnet50":
		model = ResNet(Bottleneck, [3, 4, 6, 3], grid_size, "regression", feat_nums, e_sizes, num_classes=train_props.shape[1])
	elif model_type == "densenet121":
		model = densenet121(grid_size, "regression", feat_nums, e_sizes, num_classes=train_props.shape[1])
	elif model_type == "densenet161":
		model = densenet161(grid_size, "regression", feat_nums, e_sizes, num_classes=train_props.shape[1])
	elif model_type == "densenet169":
		model = densenet169(grid_size, "regression", feat_nums, e_sizes, num_classes=train_props.shape[1])
	elif model_type == "densenet201":
		model = densenet201(grid_size, "regression", feat_nums, e_sizes, num_classes=train_props.shape[1])
	else:
		print("specify a valid model")
		return
	model.float()
	model.cuda()
	loss_function_train = nn.MSELoss(reduction='none')
	loss_function_val = nn.L1Loss(reduction='none')
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	# if model_type[0] == "r":
	# 	batch_size = 128
	# 	optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
	# 					   momentum=0.9, weight_decay=5e-4, nesterov=True)
	# elif model_type[0] == "d":
	# 	batch_size = 512
	# 	optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
	# 					   momentum=0.9, weight_decay=1e-4, nesterov=True)
	# else:
	# 	print("specify a vlid model")
	# 	return

	stds = means_stds[1, :]
	tl_list = []
	vl_list = []

	log_file = open(fn + "txt", "w")
	log_file.write("start")
	log_file.flush()

	for file_num in range(num_loads):
		if file_num % 20 == 0:
			model_file = open("../../scratch/" + fn + ".pkl", "wb")
			pickle.dump(model, model_file)
			model_file.close()

		log_file.write("load: " + str(file_num))
		print("load: " + str(file_num))
		# Get new random walks
		if file_num == 0:
			t = time.time()
			train_loader, val_loader, test_loader = get_loaders(working_dir, \
															file_num, \
															grid_size, \
															batch_size, \
															train_props, \
															val_props=val_props, \
															test_props=test_props)
			print("load time")
			print(time.time() - t)
		else:
			file_num = random.randint(0, num_walks-1)
			t = time.time()
			train_loader, _, _ = get_loaders(working_dir, \
										file_num, \
										grid_size, \
										batch_size, \
										train_props)
			print("load time")
			print(time.time() - t)
		# Train on set of random walks, can do multiple epochs if desired
		for epoch in range(epochs_per_load):
			model.train()
			t = time.time()
			train_loss_list = []
			train_mae_loss_list = []
			for i, (walks_int, walks_float, props) in enumerate(train_loader):
				walks_int = walks_int.cuda()
				walks_int = walks_int.long()
				walks_float = walks_float.cuda()
				walks_float = walks_float.float()
				props = props.cuda()
				outputs = model(walks_int, walks_float)
				# Individual losses for each item
				loss_mae = torch.mean(loss_function_val(props, outputs), 0)
				train_mae_loss_list.append(loss_mae.cpu().detach().numpy())
				loss = torch.mean(loss_function_train(props, outputs), 0)
				train_loss_list.append(loss.cpu().detach().numpy())
				# Loss converted to single value for backpropagation
				loss = torch.sum(loss)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			model.eval()
			val_loss_list = []
			with torch.no_grad():
				for i, (walks_int, walks_float, props) in enumerate(val_loader):
					walks_int = walks_int.cuda()
					walks_int = walks_int.long()
					walks_float = walks_float.cuda()
					walks_float = walks_float.float()
					props = props.cuda()
					outputs = model(walks_int, walks_float)
					# Individual losses for each item
					loss = loss_function_val(props, outputs)
					val_loss_list.append(loss.cpu().detach().numpy())
			# ith row of this array is the losses for each label in batch i
			train_loss_arr = np.array(train_loss_list)
			train_mae_arr = np.array(train_mae_loss_list)
			log_file.write("training mse loss\n")
			log_file.write(str(np.mean(train_loss_arr)) + "\n")
			log_file.write("training mae loss\n")
			log_file.write(str(np.mean(train_mae_arr)) + "\n")
			print("training mse loss")
			print(str(np.mean(train_loss_arr)))
			print("training mae loss")
			print(str(np.mean(train_mae_arr)))
			val_loss_arr = np.concatenate(val_loss_list, 0)
			val_loss = np.mean(val_loss_arr, 0)
			log_file.write("val loss\n")
			log_file.write(str(np.mean(val_loss_arr)) + "\n")
			print("val loss")
			print(str(np.mean(val_loss_arr)))
			# Unnormalized loss is for comparison to papers
			tnl = np.mean(train_mae_arr, 0)
			log_file.write("train normalized losses\n")
			log_file.write(" ".join(list(map(str, tnl))) + "\n")
			print("train normalized losses")
			print(" ".join(list(map(str, tnl))))
			log_file.write("val normalized losses\n")
			log_file.write(" ".join(list(map(str, val_loss))) + "\n")
			print("val normalized losses")
			print(" ".join(list(map(str, val_loss))))
			tunl = stds * tnl
			log_file.write("train unnormalized losses\n")
			log_file.write(" ".join(list(map(str, tunl))) + "\n")
			print("train unnormalized losses")
			print(" ".join(list(map(str, tunl))))
			vunl = stds * val_loss
			log_file.write("val unnormalized losses\n")
			log_file.write(" ".join(list(map(str, vunl))) + "\n")
			log_file.write("\n")
			print("val unnormalized losses")
			print(" ".join(list(map(str, vunl))))
			print("\n")
			print("time")
			print(time.time() - t)
		file_num += 1
		log_file.flush()
	log_file.close()
	return model

# -----------------------------------------------------   Main   ------------------------------------------------------------------------- #

# argv[1]: Directory containing walks
# argv[2]: Grid size
# argv[3]: Learning rate
# argv[4]: Batch size
# argv[5]: Number of walks to randomly select from
# argv[6]: Model type

def main():
	print(torch.cuda.is_available())
	working_dir = sys.argv[1]
	grid_size = int(sys.argv[2])
	learning_rate = float(sys.argv[3])
	batch_size = int(sys.argv[4])
	num_walks = int(sys.argv[5])
	model_type = sys.argv[6]
	fn = sys.argv[7]
	print(learning_rate)
	print(batch_size)
	train(working_dir, grid_size, learning_rate, batch_size, num_walks, model_type, fn)

if __name__ == "__main__":
	main()