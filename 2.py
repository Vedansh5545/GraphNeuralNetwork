# Load a Real Dataset
'''
This code snippet demonstrates how to load a real-world dataset using PyTorch Geometric (PyG).
It uses the Cora dataset, which is a citation network of scientific papers.
The dataset is loaded using the `Planetoid` class from the `torch_geometric.datasets` module.
'''

from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]

print(data)
