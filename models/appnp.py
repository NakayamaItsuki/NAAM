import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP as GeoAPPNP


class APPNP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, K=10, alpha=0.1, dropout=0.5, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.K = K
        self.alpha = alpha
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers-2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
        # output layer
        self.layers.append(nn.Linear(hidden_channels, out_channels))

        self.propagate = GeoAPPNP(K=10, alpha=alpha, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        x = self.propagate(x, edge_index)

        return x
