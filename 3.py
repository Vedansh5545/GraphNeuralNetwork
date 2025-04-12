# Visualize the Graph
"""
# This script demonstrates how to create a graph using PyTorch Geometric and visualize it using NetworkX and Matplotlib.
# It includes:
# 1. Creating a simple graph with 3 nodes and 2 edges.
# 2. Loading a real-world dataset (Cora).
# 3. Visualizing the graph using NetworkX and Matplotlib.
# The graph is represented using the `Data` class from PyTorch Geometric, which contains node features and edge indices.
# The Cora dataset is a citation network of scientific papers, and it is loaded using the `Planetoid` class from the 
# `torch_geometric.datasets` module.
# The graph is visualized using NetworkX and Matplotlib, with nodes colored in sky blue and labeled.
"""
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]

G = to_networkx(data, to_undirected=True)
nx.draw(G, with_labels=True, node_color='skyblue', node_size=500)
plt.show()
