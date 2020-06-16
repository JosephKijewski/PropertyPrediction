import rdkit.Chem as Chem
import numpy as np
import pickle

qm8_sdf = "qm8.sdf"
qm8_csv = "qm8.sdf.csv"
qm9_sdf = "gdb9.sdf"
qm9_csv = "gdb9.sdf.csv"
qm8_supplier = Chem.SDMolSupplier(qm8_sdf, True, False)
qm9_supplier = Chem.SDMolSupplier(qm9_sdf, True, False)

qm8_mols = []
qm9_mols = []
qm8_val_ind = []
qm9_val_ind = []
for (i, mol) in enumerate(qm8_supplier):
	if mol != None:
		valid = True
		for atom in mol.GetAtoms():
			if atom.GetDegree() == 0:
				print("bondless atom")
				valid = False
		if valid:
			qm8_mols.append(mol)
			qm8_val_ind.append(i)

for (i, mol) in enumerate(qm9_supplier):
	if mol != None:
		valid = True
		for atom in mol.GetAtoms():
			if atom.GetDegree() == 0:
				print("bondless atom")
				valid = False
		if valid:
			qm9_mols.append(mol)
			qm9_val_ind.append(i)

print(len(qm8_val_ind))
print(len(qm9_val_ind))
qm8_prop_arr = np.loadtxt(qm8_csv, delimiter=',', skiprows=1, usecols=range(1, 17))[qm8_val_ind, :]
qm9_prop_arr = np.loadtxt(qm9_csv, delimiter=',', skiprows=1, usecols=range(4, 16))[qm9_val_ind, :]

qm8_mols_fn = "../Code/data/qm8/mols.pkl"
qm8_props_fn = "../Code/data/qm8/props.csv"
qm9_mols_fn = "../Code/data/qm9/mols.pkl"
qm9_props_fn = "../Code/data/qm9/props.csv"

np.savetxt(qm8_props_fn, qm8_prop_arr, delimiter=',')
np.savetxt(qm9_props_fn, qm9_prop_arr, delimiter=',')


qm8_mols_file = open(qm8_mols_fn, "wb")
qm9_mols_file = open(qm9_mols_fn, "wb")
pickle.dump(qm8_mols, qm8_mols_file)
pickle.dump(qm9_mols, qm9_mols_file)
qm8_mols_file.close()
qm9_mols_file.close()