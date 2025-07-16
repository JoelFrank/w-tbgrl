# models/gcn_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv # Importar GCNConv de PyTorch Geometric

class GCNEncoder(nn.Module):
    def __init__(self, num_node_types, emb_dim, hidden_dim, out_dim):
        super(GCNEncoder, self).__init__()
        # La capa de embedding mapea los IDs de tipo de nodo a vectores densos
        self.embedding_layer = nn.Embedding(num_node_types, emb_dim)
        
        # Capas GCN utilizando GCNConv de PyTorch Geometric
        # GCNConv acepta edge_weight directamente
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x_node_types, edge_index, edge_weight=None):
        """
        Args:
            x_node_types (torch.Tensor): IDs de los tipos de nodos (ej. 'tipo_id' de dataset_loader.py).
            edge_index (torch.Tensor): Índices de las aristas (formato PyG).
            edge_weight (torch.Tensor, opcional): Pesos de las aristas.
        """
        # Convertir los IDs de tipo de nodo a embeddings
        x = self.embedding_layer(x_node_types)

        # Propagación a través de las capas GCN
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=self.training) # Dropout para regularización
        x = self.conv2(x, edge_index, edge_weight) # La última capa de GCN no suele tener ReLU/Dropout para la tarea downstream

        return x