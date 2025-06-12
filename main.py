import os
from metrics import evaluate_all
from utils import load_graphs

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

if __name__ == "__main__":
    sampled_g = {'GraphRNN': load_graphs('GraphRNN'),
                 #'CCGVAE': load_graphs('CCGVAE'),
                 'MolGAN': load_graphs('MolGAN'),
                 'DiGress': load_graphs('DiGress')}

    results = {}
    for name, graphs in sampled_g.items():
        print(f"\n--- Generating with {name} ---")
        metrics = evaluate_all(graphs)
        results[name] = metrics

    print("\n=== Summary of Results ===")
    for model, metrics in results.items():
        print(f"\n{model}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")