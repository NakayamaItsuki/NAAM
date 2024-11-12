import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.3, heads=8, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.dropout = dropout
        self.heads = heads

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=1, dropout=self.dropout))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=self.dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(heads*hidden_channels, hidden_channels, heads=heads, dropout=self.dropout))
            self.convs.append(GATConv(heads*hidden_channels, out_channels, heads=1, concat=False, dropout=self.dropout))
        self.dropout = nn.Dropout(p=0.6)
        
    def forward(self, x, edge_index, return_hid=False):
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            hidden_representation = x
            x = F.elu(x)
            x = self.dropout(x)
            
        x = self.convs[-1](x, edge_index)

        if return_hid:
            return x, hidden_representation
        else:
            return x
