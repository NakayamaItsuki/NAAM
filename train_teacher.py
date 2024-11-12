import argparse

import numpy as np
import torch
from torchmetrics.functional import accuracy

import datasets
from models import APPNP, GAT, GCN, GCN2, MLP, SGC
from trainer import Trainer
from utils import get_training_config, inductive_split, set_seeds


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer', choices=datasets.__all__)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.8)
    return parser.parse_args()


def main(args, **kwargs):
    device = 'cuda:' + args.device

    train_accs, val_accs, test_accs = [], [], []
    obs_test_accs, unobs_test_accs = [], []
    
    for i in range(args.n_runs):
        set_seeds(i)
        dataset_, data = getattr(datasets, args.dataset)()
        
        data = data.to(device)
        if args.mode == 'ind':
            data, unobs_test_data = inductive_split(data, args.ratio)

        n_features = dataset_.num_node_features
        n_classes = dataset_.num_classes

        if args.mode == 'ind':
            saved_path = f'saves/ind_seed_{i}_teacher_{args.model}_{args.dataset}_{args.hidden_channels}_{args.num_layers}_{args.ratio}.pt'
        else:
            saved_path = f'saves/seed_{i}_teacher_{args.model}_{args.dataset}_{args.hidden_channels}_{args.num_layers}.pt'

        dict_args = vars(args)
        if args.model == 'GCN':
            model = GCN(in_channels=n_features, hidden_channels=args.hidden_channels,
                        out_channels=n_classes, num_layers=args.num_layers, dropout=args.dropout).to(device)
        elif args.model == 'GAT':
            model = GAT(in_channels=n_features, hidden_channels=args.hidden_channels,
                        out_channels=n_classes, num_layers=args.num_layers, dropout=args.dropout).to(device)
        elif args.model == 'SGC':
            model = SGC(in_channels=n_features, out_channels=n_classes,
                        num_layers=args.num_layers, cached=False).to(device)
        elif args.model == 'GCN2':
            model = GCN2(in_channels=n_features, hidden_channels=args.hidden_channels, out_channels=n_classes,
                         num_layers=args.num_layers, dropout=args.dropout, alpha=args.alpha, theta=args.theta).to(device)
        elif args.model == 'APPNP':
            model = APPNP(in_channels=n_features, hidden_channels=args.hidden_channels,
                          out_channels=n_classes, K=args.K, alpha=args.alpha, num_layers=args.num_layers, dropout=args.dropout).to(device)
        elif args.model == 'MLP':
            model = MLP([n_features]+[args.hidden_channels]*(args.num_layers-1) +
                        [n_classes], dropout=args.dropout).to(device)
        else:
            raise NameError('Invalid model type')

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.model == 'GCN2':
            optimizer = torch.optim.Adam([
                dict(params=model.convs.parameters(), weight_decay=args.weight_decay),
                dict(params=model.lins.parameters(), weight_decay=5e-4)], lr=args.lr)

        trainer = Trainer(model, optimizer, data, device, saved_path=saved_path, patience=args.patience, ratio=args.ratio)
        trainer.train()
        
        if args.mode == 'tran':
            train_acc, val_acc, test_acc = trainer.test()
            print(f'train acc: {train_acc:.3f} | val acc: {val_acc:.3f} | test acc: {test_acc:.3f}')
            
            
        elif args.mode == 'ind':
            
            # observed test data
            train_acc, val_acc, obs_test_acc = trainer.test(args.ratio)
            
            # unobserved test data
            trainer.dataset = unobs_test_data
            pred = trainer.model(unobs_test_data.x, unobs_test_data.edge_index)
            dataset_, data = getattr(datasets, args.dataset)()
            data = data.to(device)

            print(accuracy(pred[data.test_mask], unobs_test_data.y[data.test_mask]).item())
            print(accuracy(pred[data.train_mask], unobs_test_data.y[data.train_mask]).item())
            unobs_test_acc = trainer.test()[0]
            
            # weighted sum of observed and unobserved test accuracies
            test_acc = args.ratio * obs_test_acc + (1-args.ratio) * unobs_test_acc
            
            print(f'train acc: {train_acc:.3f} | val acc: {val_acc:.3f} | obs test acc: {obs_test_acc:.3f} | unobs test acc: {unobs_test_acc:.3f} | test acc: {test_acc:.3f}')
            
            obs_test_accs.append(obs_test_acc)
            unobs_test_accs.append(unobs_test_acc)
            
            
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    if args.mode == 'tran':
        print(f'{np.mean(train_accs):.3f} {np.std(train_accs):.3f},{np.mean(val_accs):.3f} {np.std(val_accs):.3f},{np.mean(test_accs):.3f} {np.std(test_accs):.3f}')
        
    elif args.mode == 'ind':
        print(f'{np.mean(train_accs):.3f} {np.std(train_accs):.3f},{np.mean(val_accs):.3f} {np.std(val_accs):.3f},{np.mean(obs_test_accs):.3f} {np.std(obs_test_accs):.3f},{np.mean(unobs_test_accs):.3f} {np.std(unobs_test_accs):.3f}, {np.mean(test_accs):.3f} {np.std(test_accs):.3f}')
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer', choices=datasets.__all__)
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--mode', type=str, default='tran')
    parser.add_argument('--config_name', type=str, default='./train_conf.yaml')
    parser.add_argument('--device', type=str, default='-1') 
    parser.add_argument('--patience', type=int, default=50)

    temp_args, _ = parser.parse_known_args()

    if temp_args.mode == 'ind':
        parser.add_argument('--ratio', type=float, default=0.8)
    
    if temp_args.mode == 'tran':
        parser.add_argument('--ratio', type=float, default=1.0)
    
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.8)
    
    if temp_args.model == 'GCN2' or temp_args.model == 'GCN2_':
        parser.add_argument('--alpha', type=float, default=0.1)
        parser.add_argument('--theta', type=float, default=0.5)
    elif temp_args.model == 'APPNP':
        parser.add_argument('--K', type=int, default=10)
        parser.add_argument('--alpha', type=float, default=0.1)
        
    config = get_training_config(temp_args.config_name, temp_args.model, temp_args.dataset)
    parser.set_defaults(**config)
    
    args = parser.parse_args()
    print(args)

    main(args)
