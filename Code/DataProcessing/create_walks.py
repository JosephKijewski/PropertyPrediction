import csv
import sys
import numpy as np
import pickle

import rdkit.Chem as Chem

from create_walks_utils import create_all_walks

# Reads csv into an array
def read_csv(csv_file):
	csv_file = open(csv_file)
	reader = csv.reader(csv_file, delimiter=',')
	arr = []
	for row in reader:
		arr.append(row)
	arr = np.array(arr)
	return arr

# Gets mol from smiles string
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

# Creates walk files in specified directory. 
# argv[1]: Directory where data is being handled (should be the folder for the dataset)
# argv[2]: Number of threads making copies
# argv[3]: Number of copies you would like to have made per thread
# argv[4]: Grid size of each walk

def main():
	work_dir = sys.argv[1]
	num_threads = int(sys.argv[2])
	num_copies = int(sys.argv[3])
	grid_size = int(sys.argv[4])

	train_file = open(work_dir + "/train_mols.pkl", "rb")
	val_file = open(work_dir + "/val_mols.pkl", "rb")
	test_file = open(work_dir + "/test_mols.pkl", "rb")
	train_mols = pickle.load(train_file)
	val_mols = pickle.load(val_file)
	test_mols = pickle.load(test_file)
	train_file.close()
	val_file.close()
	test_file.close()

	create_all_walks(work_dir, num_threads, num_copies, grid_size, train_mols, val_mols, test_mols)

if __name__ == "__main__":
	main()