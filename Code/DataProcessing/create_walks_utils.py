import sys
import numpy as np
import csv
import os
import shutil
import time
import tracemalloc
import psutil
import pickle
from rand_walk_code import get_one_walk
from multiprocessing import Process
from torch import from_numpy
from encoding import get_feat_dict

# IMPORTANT: Get feat_dict working
def fill_int_arr(walk_arr, feat_dict):
	shape = walk_arr.shape
	int_feat_arr = np.zeros((shape[0], 5, shape[2], shape[3]), dtype=np.uint8)
	for a in range(shape[0]):
		rel_feat_dict = feat_dict[a]
		for b in range(shape[2]):
			for c in range(shape[3]):
				int_feat_arr[a, :, b, c] = int_lookup(walk_arr[a, :, b, c], rel_feat_dict)
	return int_feat_arr

def int_lookup(arr, mol_feat_dict):
	atom_feat_dict = mol_feat_dict["atom_int"]
	bond_feat_dict = mol_feat_dict["bond_int"]
	return_arr = np.zeros(5, dtype=np.uint8)
	return_arr[:4] = atom_feat_dict[arr[0]]
	return_arr[4] = bond_feat_dict[arr[1]]
	return return_arr

def float_lookup(at_index, feat_dict):
	return feat_dict[at_index]

def fill_float_arr(walk_arr, feat_dict):
	shape = walk_arr.shape
	float_feat_arr = np.zeros((shape[0], 1, shape[2], shape[3]), dtype=np.float32)
	# Why is this for loop faster than np.apply_along_axis??!?!?
	for a in range(shape[0]):
		rel_feat_dict = feat_dict[a]["bond_float"]
		for b in range(shape[2]):
			for c in range(shape[3]):
				float_feat_arr[a, 0, b, c] = float_lookup(walk_arr[a, 1, b, c], rel_feat_dict)
	return float_feat_arr

# Creates a folder with the given name, overwriting a folder if it already exists.
def make_folder(folder_name):
	if os.path.exists(folder_name):
		print("Folder overwrite occurring")
		shutil.rmtree(folder_name)
	os.mkdir(folder_name)

def create_walk_set(directory, mols, grid_size, set_num, feat_dict):
	walk_list = []
	process = psutil.Process(os.getpid())
	print(process.memory_info().rss/1024/1024/1024)
	for mol in mols:
		walk = get_one_walk(mol, grid_size)
		walk_list.append(walk)
	walk_array = np.array(walk_list, dtype=np.uint8)
	walk_array = walk_array.reshape(walk_array.shape[0], 2, grid_size, grid_size)

# correct the dictionary lookups in this file
	# add feat_dict as input to function
	int_feat_arr = fill_int_arr(walk_array, feat_dict)
	float_feat_arr = fill_float_arr(walk_array, feat_dict)
	int_feat_tensor = from_numpy(int_feat_arr)
	float_feat_tensor = from_numpy(float_feat_arr)

	fn_int = directory + "/" + str(set_num) + "i.pkl"
	fn_float = directory + "/" + str(set_num) + "f.pkl"

	file_int = open(fn_int, "wb")
	file_float = open(fn_float, "wb")

	pickle.dump(int_feat_tensor, file_int)
	pickle.dump(float_feat_tensor, file_float)

	file_int.close()
	file_float.close()

def thread(start, end, direc, mols, grid_size, feat_dict):
	for i in range(start, end):
		t = time.time()
		create_walk_set(direc, mols, grid_size, i, feat_dict)
		print(time.time() - t)

# Given the data, specifications for how many walks to make, and a location, and does the entire process of creating
# walks and their containing folders.
def create_all_walks(data_path, num_threads, num_copies, grid_size, train_mols, val_mols, test_mols):
	train_dir = data_path + "/train_walk_files" + str(grid_size)
	val_dir = data_path + "/val_walk_files" + str(grid_size)
	test_dir = data_path + "/test_walk_files" + str(grid_size)
	train_feat_dict = get_feat_dict(data_path + "/train_mols.pkl")
	val_feat_dict = get_feat_dict(data_path + "/val_mols.pkl")
	test_feat_dict = get_feat_dict(data_path + "/test_mols.pkl")
	dir_list = [train_dir, val_dir, test_dir]
	for directory in dir_list:
		make_folder(directory)
	for i in range(num_threads):
		start, end = num_copies * i, num_copies * (i+1)
		p = Process(target=thread, args=(start, end, train_dir, train_mols, grid_size, train_feat_dict))
		p.start()
	for i in range(num_threads):
		p.join()
	create_walk_set(val_dir, val_mols, grid_size, 0, val_feat_dict)
	create_walk_set(test_dir, test_mols, grid_size, 0, test_feat_dict)