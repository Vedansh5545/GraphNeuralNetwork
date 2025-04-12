from torch_geometric.datasets import Planetoid

# Load Cora dataset
dataset = Planetoid(root='data/Planetoid', name='Cora')
data = dataset[0]

"""
This code snippet demonstrates how to load a real-world dataset using PyTorch Geometric (PyG).
It uses the Cora dataset, which is a citation network of scientific papers.
The dataset is loaded using the `Planetoid` class from the `torch_geometric.datasets` module.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels*2)
        self.conv2 = GCNConv(hidden_channels*2, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)



    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x
'''This code defines a Graph Convolutional Network (GCN) model using PyTorch Geometric.
The GCN class inherits from `torch.nn.Module` and consists of two graph convolutional layers (`GCNConv`).
The `forward` method applies the first convolutional layer, applies ReLU activation, and then applies the 
second convolutional layer.
The model takes node features (`x`) and edge indices (`edge_index`) as input.
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=dataset.num_node_features,
            hidden_channels=32,
            out_channels=dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()
'''
This code sets up the training process for the GCN model.
It includes:
1. Moving the model and data to the appropriate device (GPU or CPU).
2. Initializing the optimizer (Adam) with a learning rate and weight decay.
3. Defining the `train` function, which performs one training step:
   - Sets the model to training mode.
   - Zeroes the gradients of the optimizer.
   - Computes the model's output for the training nodes.
   - Calculates the loss using cross-entropy.
   - Backpropagates the loss and updates the model parameters.'
   '''

def test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        correct = pred[mask].eq(data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        accs.append(acc)
    return accs  # [train_acc, val_acc, test_acc]
'''
This code defines the `test` function for evaluating the GCN model.
It includes:
1. Setting the model to evaluation mode.
2. Computing the model's output for all nodes.
3. Getting the predicted class for each node.
4. Calculating the accuracy for the training, validation, and test sets using the masks provided in the dataset.
The accuracies are returned as a list.
'''

for epoch in range(1, 201):
    loss = train()
    train_acc, val_acc, test_acc = test()
    if epoch % 20 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
'''
This code runs the training loop for 200 epochs.
It includes:
1. Training the model for one epoch and getting the loss.
2. Evaluating the model on the training, validation, and test sets.
3. Printing the loss and accuracies every 20 epochs.
The accuracies are printed in the format: `Epoch {epoch}, Loss: {loss}, Train Acc: {train_acc}, Val Acc: {val_acc}, Test Acc: {test_acc}`.
'''

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

@torch.no_grad()
def visualize():
    model.eval()
    out = model(data.x, data.edge_index)
    z = out.cpu().numpy()
    y = data.y.cpu().numpy()

    z_embedded = TSNE(n_components=2).fit_transform(z)

    plt.figure(figsize=(10, 7))
    plt.xticks([])
    plt.yticks([])
    scatter = plt.scatter(z_embedded[:, 0], z_embedded[:, 1], c=y, cmap='tab10', s=15)
    plt.title("Node Embeddings after GCN")
    plt.colorbar(scatter, label='Class')
    plt.show()

visualize()
'''
This code defines a function to visualize the node embeddings learned by the GCN model.
It includes:
1. Setting the model to evaluation mode.
2. Computing the model's output for all nodes.
3. Converting the output to NumPy arrays for visualization.
4. Using t-SNE to reduce the dimensionality of the embeddings to 2D.
5. Plotting the 2D embeddings using Matplotlib, with colors representing different classes.
The plot shows the node embeddings after training the GCN model.
'''