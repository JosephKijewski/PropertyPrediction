import numpy as np
import sys
import csv
import pickle
import os
import shutil
import rdkit.Chem as Chem
from sklearn.model_selection import train_test_split
from mol_feature_count import get_feature_count


hyb_dict = {Chem.rdchem.HybridizationType.SP: 0,
            Chem.rdchem.HybridizationType.SP2: 1,
            Chem.rdchem.HybridizationType.SP3: 2,
            Chem.rdchem.HybridizationType.SP3D: 3,
            Chem.rdchem.HybridizationType.SP3D2: 4,
            Chem.rdchem.HybridizationType.UNSPECIFIED: 5,
            Chem.rdchem.HybridizationType.S: 6}

bt_dict = {Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3}

# Checks whether a molecule can be valid in any dataset
def universal_check(atomic_num, degree, charge, hyb_type, num_hs):
	global mol_dict
	if degree == 0 or degree > 4:
		print("invalid degree")
		return False
	elif charge < -2 or charge > 2:
		print("invalid charge")
		return False
	elif hyb_type > 2:
		print("invalid hybridization")
		return False
	elif mol_dict["atomic num"][atomic_num] < 10:
		print("rare atom")
		return False
	elif num_hs > 3:
		print("invalid num hs")
		return False
	return True

# Determines whether or not a molecule is valid in tox21
def mol_valid_tox21(mol):
	valid = True
	for atom in mol.GetAtoms():
		atomic_num = atom.GetAtomicNum()
		degree = atom.GetDegree()
		charge = atom.GetFormalCharge()
		hyb_type = hyb_dict[atom.GetHybridization()]
		num_hs = atom.GetTotalNumHs()
		if universal_check(atomic_num, degree, charge, hyb_type, num_hs) == False:
			valid = False
		elif charge < -1 or charge > 1:
			print("invalid charge")
			valid = False
	return valid

# Determines whether or not a molecule is valid in qm8
def mol_valid_qm8(mol):
	valid = True
	for atom in mol.GetAtoms():
		atomic_num = atom.GetAtomicNum()
		degree = atom.GetDegree()
		charge = atom.GetFormalCharge()
		hyb_type = hyb_dict[atom.GetHybridization()]
		num_hs = atom.GetTotalNumHs()
		if universal_check(atomic_num, degree, charge, hyb_type, num_hs) == False:
			valid = False
		elif charge != 0:
			valid = False
	return valid

# Determines whether or not a molecule is valid in qm9
def mol_valid_qm9(mol):
	valid = True
	for atom in mol.GetAtoms():
		atomic_num = atom.GetAtomicNum()
		degree = atom.GetDegree()
		charge = atom.GetFormalCharge()
		hyb_type = hyb_dict[atom.GetHybridization()]
		num_hs = atom.GetTotalNumHs()
		if universal_check(atomic_num, degree, charge, hyb_type, num_hs) == False:
			valid = False
	return valid

# Normalize an array with column-wise means and standard deviations
def normalize_arr(arr, means, stds):
	num_mols = arr.shape[0]
	arr_mean = np.tile(means, (num_mols, 1))
	arr_std = np.tile(stds, (num_mols, 1))
	arr = (arr - arr_mean) / arr_std
	return arr

# Convert empty values to 2
def int_nan(n):
	if n == "":
		n =  2
	return int(n)

# Create processed smiles and props files in correct folder
def create_files(fold_n, save_to_folder, splits, prob_type, valid_fun):
	print("Note that nan values are not currently handled for regression datasets, and classification hasnt been verified to work")
	mols_file = open(fold_n + "/mols.pkl", "rb")
	mols_list = pickle.load(mols_file)
	mols_file.close()
	print(len(mols_list))
	valid_indices = []
	ct = 0
	for (i, mol) in enumerate(mols_list):
		try:
			mol = Chem.RemoveHs(mol)
			if valid_fun(mol):
				valid_indices.append(i)
		except Chem.rdchem.MolSanitizeException:
			ct += 1
	
	print("kekulize errors")
	print(ct)

	mols_list = [mols_list[i] for i in valid_indices]
	if prob_type == "c":
		props = np.loadtxt(fold_n + "/props.csv", dtype=int, delimiter=',')
		print(props.shape)
		# this line is not validated, not sure if it works
		props = int_nan(props)
	else:
		props = np.loadtxt(fold_n + "/props.csv", dtype=np.float32, delimiter=',')
		print(props.shape)
	props = props[valid_indices, :]

	# Split data into train, val, test sets
	x_train, x_split, y_train, y_split = train_test_split(mols_list, props, test_size=(1-splits[0]))
	x_val, x_test, y_val, y_test = train_test_split(x_split, y_split, test_size=((1-splits[0]-splits[1]) / (1-splits[0])))

	if prob_type == "r":
		train_means = np.expand_dims(np.nanmean(y_train, 0), 0)
		train_stds = np.expand_dims(np.nanstd(y_train, 0), 0)
		y_train = normalize_arr(y_train, train_means, train_stds)
		y_val = normalize_arr(y_val, train_means, train_stds)
		y_test = normalize_arr(y_test, train_means, train_stds)

	# Save new files into desired directory
	if os.path.exists(save_to_folder):
		print("Folder overwrite occurring")
		shutil.rmtree(save_to_folder)
	os.mkdir(save_to_folder)

	train_mols_fn = save_to_folder + "/train_mols.pkl"
	val_mols_fn = save_to_folder + "/val_mols.pkl"
	test_mols_fn = save_to_folder + "/test_mols.pkl"
	train_mols_file = open(train_mols_fn, "wb")
	val_mols_file = open(val_mols_fn, "wb")
	test_mols_file = open(test_mols_fn, "wb")
	pickle.dump(x_train, train_mols_file)
	pickle.dump(x_val, val_mols_file)
	pickle.dump(x_test, test_mols_file)
	train_mols_file.close()
	val_mols_file.close()
	test_mols_file.close()

	train_props_fn = save_to_folder + "/train_props.csv"
	val_props_fn = save_to_folder + "/val_props.csv"
	test_props_fn = save_to_folder + "/test_props.csv"
	
	if prob_type == "c":
		fmt = "%d"
	else:
		fmt = "%1.5f"
		np.savetxt(save_to_folder + "/means_stds.csv", np.vstack((train_means, train_stds)), delimiter=',', fmt = fmt)

	np.savetxt(train_props_fn, y_train, fmt=fmt, delimiter=',')
	np.savetxt(val_props_fn, y_val, fmt=fmt, delimiter=',')
	np.savetxt(test_props_fn, y_test, fmt=fmt, delimiter=',')

# Arguments: 
# argv[1]: path to data folder for the dataset you are using
# argv[2]: Folder where you would like to save your files to, should be unique to dataset
# argv[3]: Train percent as a float in [0, 1]
# argv[4]: Val percent as a float in [0, 1]
# argv[5]: Problem type: c, r depending on whether you want classification or regression
# argv[6]: Dataset name


def main():
	fold_n = sys.argv[1]
	save_to_folder = sys.argv[2]
	splits = (float(sys.argv[3]), float(sys.argv[4]))
	prob_type = sys.argv[5]
	dataset = sys.argv[6]
	global mol_dict
	_, mol_dict = get_feature_count(fold_n + "/mols.pkl")
	assert prob_type == "r" or prob_type == "c", "Please choose regression or classification"
	if prob_type == "c":
		print("classification")
	else:
		print("regression")
	assert((splits[0] + splits[1]) <= 1), "Training and validation ratios must add up to 1 or less"
	if dataset == "tox21":
		valid_fun = mol_valid_tox21
	elif dataset == "qm8":
		valid_fun = mol_valid_qm8
	elif dataset == "qm9":
		valid_fun = mol_valid_qm9
	create_files(fold_n, save_to_folder, splits, prob_type, valid_fun)

if __name__ == "__main__":
	main()
