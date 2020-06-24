# Code Implementation for the paper "Individualized Indicator For All: Stock-wise Technical Indicator Optimization with Stock Embedding"

import numpy as np
import pandas as pd
import networkx as nx
import random
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# Bipartite Graph Generation


df = pd.DataFrame(
    {'FUND': ['A1','A1', 'A1', 'A2', 'A2','A3','A4','A5'],
     'STOCK': ['S1', 'S2', 'S3', 'S1', 'S3', 'S4', 'S4', 'S2'],
     'WEIGHT': [10, 5, 20, 13, 19, 20, 15, 14],
     })

B = nx.Graph()
B.add_nodes_from(df['FUND'], bipartite=0)
B.add_nodes_from(df['STOCK'], bipartite=1)
B.add_weighted_edges_from(
    [(row['FUND'], row['STOCK'], row['WEIGHT']) for idx, row in df.iterrows()],
    weight='weight')

print(B.edges(data=True))


# Random Walk Sequence Generation

nx.write_adjlist(B, 'raw.adjlist')

# Stock Embedding via Skipgram
# deepwalk --input input.file --format weighted_edgelist --output output.file

###########################################################################################

def get_randomwalk(node, path_length):

    random_walk = [node]

    for i in range(path_length-1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node

    return random_walk

epoch=100
path_length = 200
all_nodes = list(G.nodes())
random_walk_seq = []

for i in range(epoch):
    idx = random.randint(0, len(all_nodes))
    initial = all_nodes[idx]

    random_walk_seq.append(get_randomwalk(node=initial, path_length=path_length))

###########################################################################################

# TTIO Model Training

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


latent_dim = 32

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        latent_dim = 32
        self.rescale = nn.linear(latent_dim, 1)

    def forward(self, input):
        out = self.rescale(input)
        out = F.softmax(out)
        return out


# Testing
