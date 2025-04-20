import torch
from torch import nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F

class WebAttackGNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_classes=6, num_heads=4):
        super(WebAttackGNN, self).__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=num_heads, dropout=0.2)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, dropout=0.2)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.fc1(x)
        embedding = F.elu(x)  # Lưu embedding
        x = self.fc2(embedding)
        return F.log_softmax(x, dim=1), embedding

    def get_embedding(self, data):
        """Trích xuất embedding từ dữ liệu"""
        self.eval()
        with torch.no_grad():
            _, embedding = self.forward(data)
        return embedding
