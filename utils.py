import pickle

def save_graphs(graphs, path):
    with open(path, "wb") as f:
        pickle.dump(graphs, f)

def load_graphs(path):
    with open(path, "rb") as f:
        return pickle.load(f)