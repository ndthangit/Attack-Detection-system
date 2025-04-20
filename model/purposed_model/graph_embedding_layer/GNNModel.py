"""GNN Model architecture with edge attribute handling"""
from torch_geometric.nn import TransformerConv
import torch.nn as nn
import torch.nn.functional as F

class GNNModel(nn.Module):
    """GNN Model architecture with edge attribute handling"""
    def __init__(self, num_features, num_classes, hidden_channels=256):
        super().__init__()
        # First convolution with edge attributes
        self.conv1 = TransformerConv(num_features, hidden_channels, heads=4, edge_dim=2)
        self.conv2 = TransformerConv(hidden_channels*4, hidden_channels, heads=4, edge_dim=2)
        self.conv3 = TransformerConv(hidden_channels*4, hidden_channels, heads=4, edge_dim=2)

        # Classification layers
        self.lin1 = nn.Linear(hidden_channels*4, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_attr, return_embeddings=False):
        # Use both edge_index and edge_attr in convolutions
        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = self.dropout(x)
        x = F.leaky_relu(self.conv3(x, edge_index, edge_attr=edge_attr))

        embeddings = F.leaky_relu(self.lin1(x))
        x = self.dropout(embeddings)
        x = self.lin2(x)

        if return_embeddings:
            return F.log_softmax(x, dim=1), embeddings
        return F.log_softmax(x, dim=1)