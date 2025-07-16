# models/predictor.py
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim): # <--- out_dim is required here
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Batch Normalization para estabilidad
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)