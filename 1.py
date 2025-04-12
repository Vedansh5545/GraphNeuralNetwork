# Understanding PyG’s Data Structure
from torch_geometric.data import Data
import torch

# Example: A graph with 3 nodes and 2 edges (0 -> 1, 1 -> 2)
edge_index = torch.tensor([[0, 1],
                           [1, 2]], dtype=torch.long)

# Each node has a feature vector of dimension 2
x = torch.tensor([[1, 2],
                  [2, 3],
                  [3, 4]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

print(data)

'''
This code creates a simple graph with 3 nodes and 2 edges using PyTorch Geometric (PyG).
The graph has 3 nodes, each with a feature vector of dimension 2. The edges are directed from 
node 0 to node 1 and from node 1 to node 2.
The `Data` object contains the node features (`x`) and the edge indices (`edge_index`).
'''