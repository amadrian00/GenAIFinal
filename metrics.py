import torch
from torch_geometric.utils import unbatch_edge_index, to_dense_adj
from rdkit import Chem
from rdkit.Chem import RWMol

# Eval helpers
def is_valid(mol):
    return mol is not None and Chem.SanitizeMol(mol, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE

def mol_from_graph(adj_matrix):
    if not isinstance(adj_matrix, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(adj_matrix)}")

    mol = RWMol()
    num_nodes = adj_matrix.size(0)

    for i in range(num_nodes):
        for j in range(i):
            if adj_matrix[i, j] > 0:
                mol.AddBond(i, j, Chem.rdchem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None


def evaluate_all(generated_graphs, test_loader):
    mols = [mol_from_graph(g) for g in generated_graphs]
    valid_mols = [m for m in mols if is_valid(m)]
    unique_mols = set(Chem.MolToSmiles(m) for m in valid_mols)

    # Metric calculations
    validity = len(valid_mols) / len(mols)
    uniqueness = len(unique_mols) / len(valid_mols) if valid_mols else 0.0

    # Novelty (naïve baseline vs test set)
    train_smiles = set()
    for batch in test_loader:
        for data in unbatch_edge_index(batch.edge_index, batch.batch):
            adj_matrix = to_dense_adj(data)

            if adj_matrix is not None:
                mol = mol_from_graph(adj_matrix)
                if mol and is_valid(mol):
                    train_smiles.add(Chem.MolToSmiles(mol))
            else:
                print(f"Skipped data with no valid adjacency matrix: {data}")

    novelty = len([s for s in unique_mols if s not in train_smiles]) / len(unique_mols) if unique_mols else 0.0

    return {
        "Validity": validity,
        "Uniqueness": uniqueness,
        "Novelty": novelty
    }