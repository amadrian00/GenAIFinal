import torch
from utils import load_smiles
from rdkit import RDLogger
from rdkit.Chem import RWMol, SanitizeMol, SanitizeFlags, rdchem, MolFromSmiles

RDLogger.DisableLog('rdApp.*')
qm9_smiles = set(load_smiles('models/ConditionalCGVAE-master/data/qm9.smi'))

# Eval helpers
def is_valid(mol):
    return mol is not None and SanitizeMol(mol, catchErrors=False) == SanitizeFlags.SANITIZE_NONE

def mol_from_graph(adj_matrix):
    if not isinstance(adj_matrix, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(adj_matrix)}")

    mol = RWMol()
    num_nodes = adj_matrix.size(0)

    for i in range(num_nodes):
        for j in range(i):
            if adj_matrix[i, j] > 0:
                mol.AddBond(i, j, rdchem.BondType.SINGLE)

    try:
        SanitizeMol(mol)
        return mol
    except:
        return None


def evaluate_all(smiles):
    valid_mols = [s for s in smiles if s !='None' and is_valid(MolFromSmiles(s))]

    unique_mols = set(smiles)

    validity = len(valid_mols) / 10000
    uniqueness = len(unique_mols) / len(valid_mols) if valid_mols else 0.0
    novelty = len([s for s in unique_mols if s not in qm9_smiles]) / len(unique_mols) if unique_mols else 0.0

    return {
        "Validity": validity,
        "Uniqueness": uniqueness,
        "Novelty": novelty
    }