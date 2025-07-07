# models/predictor.py
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim, bias=False)
        )
    def forward(self, z):
        return self.net(z)