# models/link_predictor.py
import torch.nn as nn

class LinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, z_u, z_v):
        return self.mlp(z_u * z_v).squeeze()