# models/link_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super(LinkPredictor, self).__init__()
        # Si hidden_dim no se especifica, usa un producto Hadamard simple y una capa lineal
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, 1) # Proyecta el producto Hadamard a un score escalar
        else:
            # MLP más complejo si se requiere mayor expresividad
            self.net = nn.Sequential(
                nn.Linear(in_dim * 2, hidden_dim), # Multiplica por 2 si concatenas embeddings
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.in_dim = in_dim
            self.hidden_dim = hidden_dim

    def forward(self, h_u, h_v):
        """
        Calcula un score para la existencia de un enlace entre h_u y h_v.

        Args:
            h_u (torch.Tensor): Embeddings del nodo U (batch_size, embedding_dim).
            h_v (torch.Tensor): Embeddings del nodo V (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Logits del score de enlace (batch_size, 1).
        """
        if hasattr(self, 'hidden_dim') and self.hidden_dim is not None:
            # Si se usa un MLP, concatenar embeddings
            combined_features = torch.cat([h_u, h_v], dim=1)
            score = self.net(combined_features)
        else:
            # Producto Hadamard simple seguido de capa lineal
            element_wise_product = h_u * h_v # (batch_size, embedding_dim)
            score = self.net(element_wise_product) # (batch_size, 1)

        return score