import rdkit.Chem as Chem
import csv
import sys
import pickle
from copy import deepcopy

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

def dict_add(d, name, val):
	if val in d[name]:
		d[name][val] += 1
	else:
		d[name][val] = 1

def dict_add_mol(d, name, val, mol):
	if val in d[name]:
		d[name][val].add(mol)
	else:
		d[name][val] = set()
		d[name][val].add(mol)

def get_feature_count(fn):
	ab_dict = {"atomic num": dict(), "degree": dict(), "charge": dict(), "chiral": dict(),
				"numhs": dict(), "hybrid": dict(), "aroma": dict(), "bond type": dict(), 
				"conj": dict(), "ring": dict(), "stereo": dict()}
	mol_dict = deepcopy(ab_dict)
	file = open(fn, "rb")
	mols = pickle.load(file)
	file.close()
	for mol in mols:
		for atom in mol.GetAtoms():
			dict_add(ab_dict, "atomic num", atom.GetAtomicNum())
			dict_add_mol(mol_dict, "atomic num", atom.GetAtomicNum(), mol)
			dict_add(ab_dict, "degree", atom.GetDegree())
			dict_add_mol(mol_dict, "degree", atom.GetDegree(), mol)
			dict_add(ab_dict, "charge", atom.GetFormalCharge())
			dict_add_mol(mol_dict, "charge", atom.GetFormalCharge(), mol)
			dict_add(ab_dict, "chiral", atom.GetChiralTag())
			dict_add_mol(mol_dict, "chiral", atom.GetChiralTag(), mol)
			dict_add(ab_dict, "numhs", atom.GetTotalNumHs())
			dict_add_mol(mol_dict, "numhs", atom.GetTotalNumHs(), mol)
			dict_add(ab_dict, "hybrid", hyb_dict[atom.GetHybridization()])
			dict_add_mol(mol_dict, "hybrid", hyb_dict[atom.GetHybridization()], mol)
			dict_add(ab_dict, "aroma", atom.GetIsAromatic())
			dict_add_mol(mol_dict, "aroma", atom.GetIsAromatic(), mol)
		for bond in mol.GetBonds():
			dict_add(ab_dict, "bond type", bt_dict[bond.GetBondType()])
			dict_add_mol(mol_dict, "bond type", bt_dict[bond.GetBondType()], mol)
			dict_add(ab_dict, "conj", bond.GetIsConjugated())
			dict_add_mol(mol_dict, "conj", bond.GetIsConjugated(), mol)
			dict_add(ab_dict, "ring", bond.IsInRing())
			dict_add_mol(mol_dict, "ring", bond.IsInRing(), mol)
			dict_add(ab_dict, "stereo", bond.GetStereo())
			dict_add_mol(mol_dict, "stereo", bond.GetStereo(), mol)
	print(ab_dict)
	for a in mol_dict:
		for b in mol_dict[a]:
			mol_dict[a][b] = len(mol_dict[a][b])
	print(mol_dict)
	return ab_dict, mol_dict
	
def main():
	fn = sys.argv[1]
	get_feature_count(fn)


if __name__ == "__main__":
	main()
