import os
from utils import load_graphs
from metrics import evaluate_all
from rdkit.Chem import SDMolSupplier, MolToSmiles

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

if __name__ == "__main__":
    qm9 = SDMolSupplier('models/DiGress-main/data/qm9/qm9_pyg/raw/gdb9.sdf', removeHs=False)
    qm9_smiles = [MolToSmiles(mol) for mol in qm9 if mol is not None]
    qm9_smiles = set(qm9_smiles)


    sampled_g = {'GraphRNN': load_graphs('GraphRNN'),
                 'CCGVAE': load_graphs('CCGVAE'),
                 'MolGAN': load_graphs('MolGAN'),
                 'DiGress': load_graphs('DiGress')}

    results = {}
    for name, graphs in sampled_g.items():
        print(f"\n--- Generating with {name} ---")
        metrics = evaluate_all(graphs, qm9_smiles)
        results[name] = metrics

    print("\n=== Summary of Results ===")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")