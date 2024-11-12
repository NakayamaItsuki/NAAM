from typing import Tuple, Union

import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import NodeStorage
from torch_geometric.datasets import Amazon, Planetoid
from torch_geometric.utils import to_undirected

__all__ = ['cora', 'citeseer', 'pubmed', 'amazon_photo', 'amazon_computers']

class NodeSplit(T.RandomNodeSplit):
    def __init__(self, num_val_per_class=30, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_val_per_class = num_val_per_class

    def _split(self, store: NodeStorage) -> Tuple[Tensor, Tensor, Tensor]:
        num_nodes = store.num_nodes

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        y = getattr(store, self.key)
        num_classes = int(y.max().item()) + 1
        for c in range(num_classes):
            idx = (y == c).nonzero(as_tuple=False).view(-1)
            idx = idx[torch.randperm(idx.size(0))]
            train_idx = idx[:self.num_train_per_class]
            val_idx = idx[self.num_train_per_class:self.num_train_per_class+self.num_val_per_class]
            train_mask[train_idx] = True
            val_mask[val_idx] = True

        remaining = (~torch.logical_or(train_mask, val_mask)).nonzero(as_tuple=False).view(-1)
        test_mask[remaining] = True
        
        return train_mask, val_mask, test_mask


class BinarizeFeatures(T.BaseTransform):
    def __call__(self, data: Union[Data, HeteroData]):
        data.x[data.x != 0] = 1
        return data


def amazon_computers():
    transforms = T.Compose([T.LargestConnectedComponents(),
                            NodeSplit()])
    dataset = Amazon('/tmp', name='computers', transform=transforms)
    data = dataset[0]
    return dataset, data


def amazon_photo():
    transforms = T.Compose([T.LargestConnectedComponents(),
                            NodeSplit()])
    dataset = Amazon('/tmp', name='photo', transform=transforms)
    data = dataset[0]
    return dataset, data


def cora():
    dataset = Planetoid('/tmp', name='cora',
                        transform=T.Compose([T.LargestConnectedComponents(),
                                             NodeSplit()]))
    data = dataset[0]
    return dataset, data


def citeseer():
    dataset = Planetoid('/tmp', name='citeseer',
                        transform=T.Compose([T.LargestConnectedComponents(),
                                             NodeSplit()]))
    data = dataset[0]
    return dataset, data


def pubmed():
    dataset = Planetoid('/tmp', name='pubmed',
                        transform=T.Compose([T.LargestConnectedComponents(),
                                             BinarizeFeatures(),
                                             NodeSplit()]))
    data = dataset[0]
    return dataset, data

