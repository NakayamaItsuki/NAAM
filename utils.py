import random
from typing import Tuple

import numpy as np
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def set_seeds(seed):
    """ Set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_training_config(config_path, model_name=None, dataset=None):
    with open(config_path, 'r') as conf:
        full_config = yaml.load(conf, Loader=yaml.FullLoader)
    dataset_specific_config = full_config['global']
    if model_name is None or dataset is None or model_name not in full_config[dataset]:
        return dataset_specific_config

    model_specific_config = full_config[dataset][model_name]

    if model_specific_config is not None:
        specific_config = dict(dataset_specific_config, **model_specific_config)
    else:
        specific_config = dataset_specific_config

    specific_config['model_name'] = model_name
    return specific_config


def idx_split(tensor: torch.Tensor, ratio=0.8):
    n = tensor.size(0)
    perm = torch.randperm(n)
    idx = int(n*ratio)
    idx_tran = tensor[perm[:idx]]
    idx_ind = tensor[perm[idx:]]
    return idx_tran, idx_ind


def inductive_split(data: Data, ratio=0.8) -> Tuple[Data, Data]:
    idx_train = data.train_mask.nonzero().flatten()
    idx_val = data.val_mask.nonzero().flatten()
    idx_test = data.test_mask.nonzero().flatten()

    idx_tran, idx_ind = idx_split(idx_test, ratio=ratio)

    idx_obs = torch.cat([idx_train, idx_val, idx_tran]).sort()[0]
    obs_edge_index = subgraph(idx_obs, data.edge_index, relabel_nodes=True)[0]

    tran_test_mask = data.test_mask.clone()
    tran_test_mask[idx_ind] = False
    ind_test_mask = data.test_mask.clone()
    ind_test_mask[idx_tran] = False
    train_data = Data(x=data.x[idx_obs], edge_index=obs_edge_index, y=data.y[idx_obs],
                      train_mask=data.train_mask[idx_obs], val_mask=data.val_mask[idx_obs], test_mask=tran_test_mask[idx_obs])
    test_data = Data(x=data.x, edge_index=data.edge_index, y=data.y, test_mask=ind_test_mask)
    return train_data, test_data


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        
        self.val_loss_min = val_loss
