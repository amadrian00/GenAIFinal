import pickle
import random
from rdkit.Chem.rdmolfiles import MolToSmiles
from rdkit.Chem import RWMol, Atom, BondType, SanitizeMol


def load_smiles(path):
    smiles = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                smiles.append(parts[0])
    return smiles

def load_graphs(model):
    res = None
    if model=='GraphRNN':
        with open('models/graph-generation-master/graphs/GraphRNN_RNN_qm9_4_128_pred_1000_1.dat', "rb") as f:
            graphs = pickle.load(f)

        res = []
        for G in graphs:
            mol = RWMol()
            node_idx_map = {}

            for node in G.nodes:
                atom = Atom('C')
                idx = mol.AddAtom(atom)
                node_idx_map[node] = idx

            for u, v in G.edges():
                mol.AddBond(node_idx_map[u], node_idx_map[v], BondType.SINGLE)

            mol = mol.GetMol()
            try:
                SanitizeMol(mol)
                res.append(MolToSmiles(mol))
            except ValueError:
                pass

    #elif model=='CCGVAE':
        #path='/models/ConditionalCGVAE-master'
        #total = 0

    elif model=='MolGAN':
        res = load_smiles('models/MolGAN/graphs/generated_molecules.smi')

    elif model=='DiGress':
        res = load_smiles('models/DiGress-main/outputs/2025-06-09/00-46-36-graph-tf-model/final_smiles.txt')

    return res

