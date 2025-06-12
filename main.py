import os
from metrics import evaluate_all
from utils import load_graphs

out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

def format_row_centered(r):
    return " | ".join(str(cell).center(width) for cell, width in zip(r, col_widths))

if __name__ == "__main__":
    sampled_g = {'GraphRNN': load_graphs('GraphRNN'),
                 #'CCGVAE': load_graphs('CCGVAE'),
                 'MolGAN': load_graphs('MolGAN'),
                 'DiGress': load_graphs('DiGress')}

    results = {}
    for name, graphs in sampled_g.items():
        metrics = evaluate_all(graphs)
        results[name] = metrics

    metric_names = list(next(iter(results.values())).keys())

    header = ["Model"] + metric_names

    rows = []
    for model_name, metrics in results.items():
        row = [model_name] + [f"{metrics[k]:.4f}" for k in metric_names]
        rows.append(row)

    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*([header] + rows))]

    print(format_row_centered(header))
    print("-+-".join("-" * width for width in col_widths))
    for row in rows:
        print(format_row_centered(row))