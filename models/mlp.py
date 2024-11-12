from typing import List

import torch
from torch_geometric.nn import MLP as GeoMLP


class MLP(torch.nn.Module):
    def __init__(self, channel_list: List[int], dropout=0.8, **kwargs):
        super().__init__()
        self.hidden_channels = channel_list[1]
        self.num_layers = len(channel_list) - 1
        self.model = GeoMLP(channel_list, dropout=dropout, batch_norm=False)

    def forward(self, x, *args, **kwargs):
        x = self.model(x)
        return x
