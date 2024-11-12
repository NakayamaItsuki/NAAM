import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCN2Conv


class GCN2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha=0.1, theta=0.5,
                 shared_weights=True, dropout=0.5, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GCN2Conv(hidden_channels, alpha, theta, layer + 1, shared_weights))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        x = self.dropout(x)
        x = x_0 = self.lins[0](x).relu()

        for conv in self.convs:
            x = self.dropout(x)
            x = conv(x, x_0, edge_index)
            x = x.relu()

        x = self.dropout(x)
        x = self.lins[1](x)

        return x
