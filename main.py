from models_wrapper import load_models
from data import load_molecular_dataset
from metrics import evaluate_all
from utils import save_graphs

import torch
import os

SEED = 42
dataset_name = "ZINC"
n_samples = 1000
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(SEED)

data = load_molecular_dataset(dataset_name)

models = load_models()
results = {}
for name, model in models.items():
    print(f"\n--- Generating with {name} ---")
    generated_graphs = model.generate(n_samples)
    save_graphs(generated_graphs, f"{out_dir}/{name}_gen.pkl")
    metrics = evaluate_all(generated_graphs, data['test'])
    results[name] = metrics

print("\n=== Summary of Results ===")
for model, metrics in results.items():
    print(f"\n{model}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")