import random
import numpy as np
import os, random
import time
import rdkit.Chem as Chem
from rdkit.Chem.AllChem import EmbedMolecule

# -----------------------------------------------------   Random walk functions   --------------------------------------------------------- #

# Note on time: Almost all time is in the loops as expected, with both loops taking almost the exact
# same amount of time (0.002 seconds each per walk for 16x16, total walk is about 0.005 seconds)
# Should output a list of features that can be reshaped to (2, n, n)
def get_one_walk(mol, n):
    neighbor_dict = dict()
    n_atoms = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        neighbor_dict[atom.GetIdx()] = set()
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        x = a1.GetIdx()
        y = a2.GetIdx()
        neighbor_dict[x].add(y)
        neighbor_dict[y].add(x)
    return rand_walk(n_atoms, neighbor_dict, n, mol)

def rand_walk(n_atoms, neighbor_dict, n, mol):
    # Prepare random walk structure
    grid = np.zeros((n, n, 2))
    grid.fill(None)
    init_atom = random.randint(0, n_atoms-1)
    if len(neighbor_dict[init_atom]) == 0:
        print(Chem.rdmolfiles.MolToSmiles(mol))
    random_neighbor = random.sample(neighbor_dict[init_atom], 1)[0]
    grid[0, 0, 0] = init_atom
    grid[0, 0, 1] = mol.GetBondBetweenAtoms(init_atom, random_neighbor).GetIdx()
    to_expand = [(0, 0, init_atom)]
    while (len(to_expand) > 0):
        expand_triple = to_expand.pop(random.randint(0, len(to_expand)-1))
        i = expand_triple[0]
        j = expand_triple[1]
        current_atom = expand_triple[2]
        neighbors = neighbor_dict[current_atom]
        if (len(neighbors) >= 2):
            expansions = random.sample(neighbors, 2)
            lower_update_atom = expansions[0]
            right_update_atom = expansions[1]
        else:
            expansion = random.sample(neighbors, 1)[0]
            lower_update_atom = expansion
            right_update_atom = expansion
        if (i+1)<n and np.isnan(grid[i+1, j, 0]):
            grid[i+1, j, 0] = lower_update_atom
            grid[i+1, j, 1] = mol.GetBondBetweenAtoms(lower_update_atom, current_atom).GetIdx()
            to_expand.append((i+1, j, lower_update_atom))
        if (j+1)<n and np.isnan(grid[i, j+1, 0]):
            grid[i, j+1, 0] = right_update_atom
            grid[i, j+1, 1] = mol.GetBondBetweenAtoms(right_update_atom, current_atom).GetIdx()
            to_expand.append((i, j+1, right_update_atom))
    grid = grid.transpose(2, 0, 1)
    grid = grid.reshape(np.prod(grid.shape),)
    grid.tolist()
    return grid

import time
mol = Chem.MolFromSmiles("[H]C([H])([H])C([H])([H])[C@@]1([H])C([H])([H])C([H])([H])C12C([H])([H])C2([H])[H]")

timer = time.time()
get_one_walk(mol, 64)
print(time.time() - timer)
# Feature frequencies:

# tox21
# {'atomic num': {6: 7751, 8: 6450, 7: 4601, 16: 1345, 15: 238, 17: 1221, 53: 82, 30: 9, 9: 547, 20: 4, 33: 9, 35: 213, 5: 26, 1: 2, 19: 3, 14: 64, 29: 5, 12: 2, 80: 14, 24: 5, 40: 2, 50: 19, 11: 8, 56: 4, 79: 5, 46: 1, 81: 1, 26: 7, 13: 8, 64: 2, 47: 1, 42: 1, 23: 1, 60: 1, 27: 4, 70: 1, 82: 1, 51: 3, 49: 3, 3: 1, 28: 4, 83: 2, 48: 3, 22: 3, 34: 6, 66: 1, 25: 3, 38: 1, 4: 1, 78: 3, 32: 1}, 
# 'degree': {1: 7622, 2: 7585, 3: 7180, 4: 2557, 0: 57, 5: 1, 6: 1}, 
# 'charge': {0: 7810, -1: 845, 2: 34, 1: 658, 3: 8, -2: 1}, 
# 'chiral': {rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 7831, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 1094, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1133}, 
# 'numhs': {3: 5303, 2: 5908, 0: 7613, 1: 7006, 4: 1, 6: 1}, 
# 'hybrid': {2: 7334, 1: 7009, 6: 36, 0: 324, 4: 9, 3: 13, 5: 7}, 
# 'aroma': {False: 7785, True: 5038}, 
# 'bond type': {0: 7743, 3: 5038, 1: 5581, 2: 270}, 
# 'conj': {False: 7337, True: 6531}, 
# 'ring': {False: 7725, True: 6050}, 
# 'stereo': {rdkit.Chem.rdchem.BondStereo.STEREONONE: 7812, rdkit.Chem.rdchem.BondStereo.STEREOZ: 169, rdkit.Chem.rdchem.BondStereo.STEREOE: 333}}

# qm8
# {'atomic num': {6: 21711, 7: 12611, 8: 17790, 9: 308, 1: 178}, 
# 'degree': {1: 20133, 2: 21448, 3: 19876, 4: 6372}, 
# 'charge': {0: 21712, 1: 21, -1: 21}, 
# 'chiral': {rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 21712, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 9238, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 8285}, 
# 'numhs': {1: 20437, 0: 20791, 2: 17637, 3: 12455}, 
# 'hybrid': {0: 5143, 1: 12513, 2: 20489, 6: 178}, 
# 'aroma': {False: 21583, True: 3636}, 
# 'bond type': {2: 5143, 1: 9830, 0: 21517, 3: 3636}, 
# 'conj': {False: 20565, True: 8717}, 
# 'ring': {False: 20401, True: 18102}, 
# 'stereo': {rdkit.Chem.rdchem.BondStereo.STEREONONE: 21712, rdkit.Chem.rdchem.BondStereo.STEREOE: 545, rdkit.Chem.rdchem.BondStereo.STEREOZ: 277}} 

# qm9
# {'atomic num': {6: 132403, 7: 82348, 8: 112469, 1: 1554, 9: 2161}, 
# 'degree': {1: 124263, 2: 131372, 3: 125897, 4: 52358}, 
# 'charge': {0: 132404, 1: 22997, -1: 9876}, 
# 'chiral': {rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 132404, rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 3}, 
# 'numhs': {1: 128404, 0: 127427, 2: 113110, 3: 79401}, 
# 'hybrid': {0: 32205, 1: 80706, 2: 127699, 6: 1554}, 
# 'aroma': {False: 132009, True: 21969}, 
# 'bond type': {2: 32151, 1: 63911, 0: 131878, 3: 21969}, 
# 'conj': {False: 128069, True: 51726}, 
# 'ring': {False: 125704, True: 118291}, 
# 'stereo': {rdkit.Chem.rdchem.BondStereo.STEREONONE: 132404, rdkit.Chem.rdchem.BondStereo.STEREOZ: 1056, rdkit.Chem.rdchem.BondStereo.STEREOE: 2342}}