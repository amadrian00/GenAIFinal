import torch
from rdkit.Chem import RWMol, SanitizeMol, SanitizeFlags, rdchem, MolToSmiles

# Eval helpers
def is_valid(mol):
    return mol is not None and SanitizeMol(mol, catchErrors=True) == SanitizeFlags.SANITIZE_NONE

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


def evaluate_all(generated_graphs, qm9_smiles):
    mols = [mol_from_graph(g) for g in generated_graphs]
    valid_mols = [m for m in mols if is_valid(m)]
    unique_mols = set(MolToSmiles(m) for m in valid_mols)

    validity = len(valid_mols) / len(mols)
    uniqueness = len(unique_mols) / len(valid_mols) if valid_mols else 0.0
    novelty = len([s for s in unique_mols if s not in qm9_smiles]) / len(unique_mols) if unique_mols else 0.0

    return {
        "Validity": validity,
        "Uniqueness": uniqueness,
        "Novelty": novelty
    }