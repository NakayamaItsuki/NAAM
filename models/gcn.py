import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels, **kwargs))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels, **kwargs))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, **kwargs))
            self.convs.append(GCNConv(hidden_channels, out_channels, **kwargs))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](x, edge_index)

        return x
