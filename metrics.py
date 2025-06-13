import torch
from utils import load_smiles
from rdkit import RDLogger
from rdkit.Chem import RWMol, SanitizeMol, SanitizeFlags, rdchem, MolFromSmiles

RDLogger.DisableLog('rdApp.*')
qm9_smiles = set(load_smiles('models/ConditionalCGVAE-master/data/qm9.smi', original=True))

# Eval helpers
def is_valid(mol):
    """
    Checks whether a molecule is valid or not with rdkit library (all logs are dumped)
    """
    return mol is not None and SanitizeMol(mol, catchErrors=False) == SanitizeFlags.SANITIZE_NONE

def mol_from_graph(adj_matrix):
    """
    Takes nx graphs and turns them into rdkit.Chem.rdchem.Mol.
    """
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
    """
    Evaluates the validity, uniqueness and novelty of the given list of smiles for the QM9 training set.
    """
    valid_mols = [s for s in smiles if s !='None' and is_valid(MolFromSmiles(s))]
    unique_mols = set(valid_mols)

    validity = len(valid_mols) / 10000
    uniqueness = len(unique_mols) / len(valid_mols) if len(valid_mols)!=0 else 0.0

    novel = len(unique_mols-qm9_smiles)
    novelty = novel / len(unique_mols) if len(unique_mols)!=0 else 0.0
    novelty_total = novel / 10000


    return {
        "Validity": validity,
        "Uniqueness": uniqueness,
        "Novelty": novelty,
        "Novelty_total": novelty_total
    }