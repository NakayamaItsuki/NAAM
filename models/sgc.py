import torch
from torch_geometric.nn import SGConv


class SGC(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        
        self.conv1 = SGConv(in_channels, out_channels, K=num_layers, **kwargs)

    def forward(self, x, edge_index):
        return self.conv1(x, edge_index)
