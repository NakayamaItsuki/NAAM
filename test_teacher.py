import argparse

import numpy as np
import torch
from torch_geometric import seed_everything
from torchmetrics.functional import accuracy

import datasets
from models import GAT, GCN, GCN2, SGC, APPNP, MLP
from utils import get_training_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer', choices=datasets.__all__)
    parser.add_argument('--teacher_model', type=str, default='GCN')
    parser.add_argument('--teacher_hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--device', type=int, default=-1)
    return parser.parse_args()


def main(dataset: str, teacher_model: str, teacher_hidden_channels: int, n_runs: int, num_layers=2, **kwargs):
    device = 'cuda:' + str(args.device)

    train_accs, val_accs, test_accs = [], [], []
    for i in range(n_runs):
        seed_everything(i)
        dataset_, data = getattr(datasets, dataset)()
        data = data.to(device)
        n_features = dataset_.num_node_features
        n_classes = dataset_.num_classes

        saved_path = f'saves/seed_{i}_teacher_{teacher_model}_{dataset}_{teacher_hidden_channels}_{num_layers}.pt' # 変更点

        if teacher_model == 'GCN':
            model = GCN(n_features, hidden_channels=teacher_hidden_channels,
                        out_channels=n_classes, num_layers=num_layers, dropout=0).to(device)
        elif teacher_model == 'GAT':
            model = GAT(n_features, hidden_channels=teacher_hidden_channels,
                        out_channels=n_classes, num_layers=num_layers, dropout=0).to(device)
        elif teacher_model == 'SGC':
            model = SGC(in_channels=n_features, out_channels=n_classes, num_layers=num_layers, cached=True).to(device)
        elif teacher_model == 'GCN2':
            model = GCN2(in_channels=n_features, hidden_channels=teacher_hidden_channels,
                         out_channels=n_classes, num_layers=num_layers).to(device)
        elif teacher_model == 'APPNP':
            config = get_training_config('./train_conf.yaml', 'APPNP', dataset)
            model = APPNP(in_channels=n_features, hidden_channels=teacher_hidden_channels,
                          out_channels=n_classes, K=config['K'], alpha=config['alpha'], num_layers=num_layers).to(device)
        elif teacher_model == 'MLP':
            model = MLP([n_features]+[teacher_hidden_channels]*(num_layers-1) + [n_classes]).to(device)
        else:
            raise NameError('Invalid model type')
        model.load_state_dict(torch.load(saved_path))

        model.eval()
        pred = model(data.x, data.edge_index).argmax(dim=-1)

        accs = []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            accs.append(accuracy(pred[mask], data.y[mask]).item())
        train_acc, val_acc, test_acc = accs
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        print(f'train acc: {train_acc:.3f} | val acc: {val_acc:.3f} | test acc: {test_acc:.3f}')

    print(f'{np.mean(train_accs):.3f} {np.std(train_accs):.3f},{np.mean(val_accs):.3f} {np.std(val_accs):.3f},{np.mean(test_accs):.3f} {np.std(test_accs):.3f}')


if __name__ == '__main__':
    args = get_args()
    dict_args = vars(args)
    main(**dict_args)
