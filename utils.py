import pickle
import networkx as nx

def save_graphs(graphs, path):
    with open(path, "wb") as f:
        pickle.dump(graphs, f)

def load_graphs(model):
    if model=='GraphRNN':
        path='/models/graph-generation-master/graphs/'

    elif model=='CCGVAE':
        path='/models/ConditionalCGVAE-master'

    elif model=='MolGAN':
        path='/models/MolGAN/'

    elif model=='DiGress':
        path='/models/DiGress/'

    else:
        return None
    with open(path, "rb") as f:
        return pickle.load(f)