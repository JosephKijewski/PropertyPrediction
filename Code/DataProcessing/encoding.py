import rdkit.Chem as Chem
import numpy as np
import pandas as pd

import time
import pickle

# -----------------------------------------------------   Encoding Functions   --------------------------------------------------------- #

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

# # Function for getting edge distances (poached from https://chemistry.stackexchange.com/questions/119955/euclidean-distance-between-atoms-using-rdkit)
# def get_edge_distance(mol_conf, bond, atom_dict):
#     if mol_conf == -1:
#         # If there is no valid mol_conf, which is a true for a very small number of molecules,
#         # what do we want to return?
#         return something
#     begin_atom = bond.GetBeginAtomIdx()
#     end_atom = bond.GetEndAtomIdx()
#     if begin_atom in atom_dict:
#         at1Coords = atom_dict[begin_atom]
#     else:
#         at1Coords = np.array(mol_conf.GetAtomPosition(begin_atom))
#         atom_dict[begin_atom] = at1Coords
#     if end_atom in atom_dict:
#         at2Coords = atom_dict[end_atom]
#     else:
#         at2Coords = np.array(mol_conf.GetAtomPosition(end_atom))
#         atom_dict[end_atom] = at2Coords
#     return np.linalg.norm(at2Coords - at1Coords)

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

orbital_multipliers = []
orbital_class_nums = [4, 4, 3, 3]
curr_mult = 1
for i in range(len(orbital_class_nums)):
    orbital_multipliers = [curr_mult] + orbital_multipliers
    curr_mult = curr_mult * orbital_class_nums[-i-1]

bond_multipliers = []
bond_class_nums = [4, 2, 2, 6]
curr_mult = 1
for i in range(len(bond_class_nums)):
    bond_multipliers = [curr_mult] + bond_multipliers
    curr_mult = curr_mult * bond_class_nums[-i-1]


def encode_features(vals, multipliers):
	# Note: I know this looks bad but it runs ~60% faster than a for loop doing the same thing
	encoding = vals[0] * multipliers[0] + vals[1] * multipliers[1] + vals[2] * multipliers[2] + vals[3] * multipliers[3]
	return encoding

def encode_bond(bond):
	bt, conj, ring, stereo = bt_dict[bond.GetBondType()], bond.GetIsConjugated(), bond.IsInRing(), bond.GetStereo()
	return encode_features([bt, conj, ring, stereo], bond_multipliers)

def encode_atom(atom):
	atomic_num = atom.GetAtomicNum()
	degree, num_hs, chiral, hyb = atom.GetDegree()-1, atom.GetTotalNumHs(), int(atom.GetChiralTag()), hyb_dict[atom.GetHybridization()]
	encoding_arr = np.zeros(4, dtype=int)
	encoding_arr[0] = atomic_num
	encoding_arr[1] = encode_features([degree, num_hs, chiral, hyb], orbital_multipliers) 
	encoding_arr[2] = atom.GetFormalCharge() + 2
	encoding_arr[3] = atom.GetIsAromatic()
	return encoding_arr

def get_mol_feats(mol):
	feat_dict = dict()
	atom_feat_arr = np.zeros((mol.GetNumAtoms(), 4))
	bond_feat_float_arr = np.zeros(mol.GetNumBonds())
	bond_feat_int_arr = np.zeros(mol.GetNumBonds())
	for atom in mol.GetAtoms():
		atom_idx = atom.GetIdx()
		atom_feat_arr[atom_idx, :] = encode_atom(atom)
	t = time.time()
	conf = mol.GetConformer()
	for bond in mol.GetBonds():
		bond_idx = bond.GetIdx()
		bond_feat_int_arr[bond_idx] = encode_bond(bond)
		bond_feat_float_arr[bond_idx] = Chem.rdMolTransforms.GetBondLength(conf, bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
	feat_dict["atom_int"] = atom_feat_arr.astype(np.uint8)
	feat_dict["bond_float"] = bond_feat_float_arr
	feat_dict["bond_int"] = bond_feat_int_arr.astype(np.uint8)
	return feat_dict

def get_feat_dict(path):
	# Ensure astype is necessary
	# Check dimensionality
	file = open(path, "rb")
	mol_list = pickle.load(file)
	file.close()
	feat_dict = dict()
	for (i, mol) in enumerate(mol_list):
		feat_dict[i] = get_mol_feats(mol)
	return feat_dict