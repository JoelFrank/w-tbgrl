# models/gcn_encoder.py
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNEncoder(nn.Module):
    def __init__(self, num_node_types, emb_dim, hidden_dim, out_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_node_types, emb_dim)
        self.conv1 = GCNConv(emb_dim, hidden_dim, bias=False)
        self.conv2 = GCNConv(hidden_dim, out_dim, bias=False)

    def forward(self, edge_index, tipo_ids, mask_embed):
        x = self.embedding(tipo_ids)
        x = x * mask_embed.unsqueeze(1)
        h = F.relu(self.conv1(x, edge_index))
        return self.conv2(h, edge_index)