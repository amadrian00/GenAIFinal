import networkx as nx
import numpy as np

import torch_geometric
from torch_geometric.utils import to_networkx

from utils import *
from data import *

def create(args):
### load datasets
    graphs=[]
    # synthetic graphs
    if args.graph_type=='grid':
        graphs = []
        for i in range(10,20):
            for j in range(10,20):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 40
    elif args.graph_type == 'qm9':
        qm9_dataset = torch_geometric.datasets.QM9(root='./data/QM9')
        processed_graphs = []
        for data in qm9_dataset:
            G = to_networkx(data, to_undirected=True)
            if nx.is_connected(G):
                G = nx.convert_node_labels_to_integers(G)
                if (G.number_of_nodes() >= 4) and (G.number_of_nodes() <= 29):
                    processed_graphs.append(G)
        shuffle(processed_graphs)
        graphs = processed_graphs
        args.max_prev_node = 25


    return graphs


