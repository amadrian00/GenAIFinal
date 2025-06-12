import pickle
import networkx as nx

def save_graphs(graphs, path):
    with open(path, "wb") as f:
        pickle.dump(graphs, f)

def load_graphs(model):
    if model=='GraphRNN':
        path='/models/graph-generation-master/graphs/'
        total = 0

    elif model=='CCGVAE':
        path='/models/ConditionalCGVAE-master'
        total = 0

    elif model=='MolGAN':
        path='/models/MolGAN/graphs/generated_molecules.sdf'
        total = 10000

    elif model=='DiGress':
        path='/models/DiGress/outputs/2025-06-09/00-46-36-graph-tf-model/final_smiles.txt'
        total = 10000

    else:
        return None

    with open(path, "rb") as f:
        return pickle.load(f), total