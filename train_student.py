import logging
from argparse import ArgumentParser
from collections import ChainMap
from contextlib import redirect_stdout
from pathlib import Path
from time import time
from typing import List
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric import seed_everything
from torch_geometric.loader.dataloader import DataLoader

import datasets
from lit_module import LitDistilledModel, LitGLNN
from lit_module import LitGLNN
from models import APPNP, GAT, GCN, GCN2, MLP, SGC
from utils import get_training_config, inductive_split, set_seeds

logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)


teachers = []

def load_teachers(teacher_models_name: List = ['GCN'], teacher_layers: List = [2], teacher_hidden_channels: List = [64], dataset: str = 'citeseer', *,
                    n_features, n_classes, seed, **kwargs):
    assert(len(teacher_models_name) == len(teacher_layers) and len(teacher_layers) == len(teacher_hidden_channels))

    if len(teachers) == 0: 
        for model_name, num_layers, hidden_channels in zip(teacher_models_name, teacher_layers, teacher_hidden_channels):
            if model_name == 'GCN':
                teachers.append(GCN(n_features, hidden_channels=hidden_channels,
                                out_channels=n_classes, num_layers=num_layers))
            elif model_name == 'GAT':
                teachers.append(GAT(n_features, hidden_channels=hidden_channels,
                                out_channels=n_classes, num_layers=num_layers))
            elif model_name == 'SGC':
                teachers.append(SGC(in_channels=n_features, out_channels=n_classes, num_layers=num_layers, cached=False))
            elif model_name == 'GCN2':
                teachers.append(GCN2(in_channels=n_features, hidden_channels=hidden_channels,
                                out_channels=n_classes, num_layers=num_layers))
            elif model_name == 'APPNP':
                config = get_training_config('./train_conf.yaml', 'APPNP', dataset)
                teachers.append(APPNP(in_channels=n_features, hidden_channels=hidden_channels,
                                out_channels=n_classes, num_layers=num_layers, K=config['K'], alpha=config['alpha']))
            elif model_name == 'MLP':
                teachers.append(MLP([n_features]+[hidden_channels]*(args.num_layers-1)+[n_classes]))
            else:
                raise NameError('Invalid model type')

    for teacher in teachers:
        model_name = type(teacher).__name__
        
        device = 'cuda:'+kwargs['device']
        if kwargs['mode'] == 'tran':

            if isinstance(teacher, SGC):
                teacher.load_state_dict(torch.load(
                    f'saves/seed_{seed}_teacher_{model_name}_{dataset}_64_{teacher.num_layers}.pt', map_location=torch.device(device)))
            else:
                teacher.load_state_dict(torch.load(
                    f'saves/seed_{seed}_teacher_{model_name}_{dataset}_{teacher.hidden_channels}_{teacher.num_layers}.pt', map_location=torch.device(device)))
        
        elif kwargs['mode'] == 'ind':

            if isinstance(teacher, SGC):
                teacher.load_state_dict(torch.load(
                    f'saves/ind_seed_{seed}_teacher_{model_name}_{dataset}_64_{teacher.num_layers}_{args.ratio}.pt', map_location=torch.device(device)))
            else:
                teacher.load_state_dict(torch.load(
                    f'saves/ind_seed_{seed}_teacher_{model_name}_{dataset}_{teacher.hidden_channels}_{teacher.num_layers}_{args.ratio}.pt', map_location=torch.device(device))) 

        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
    return nn.ModuleList(teachers)


def get_loggers(args):
    csv_logger = CSVLogger(save_dir=args.save_dir, name=args.exp_group_name)
    return [csv_logger]


def create_model(*args, **kwargs) -> pl.LightningModule:
    if kwargs['method'] == 'attn':
        model = LitDistilledModel(**kwargs)
    elif kwargs['method'] == 'glnn':
        model = LitGLNN(**kwargs)
    else:
        raise NameError('Invalid method')
    return model


def train(args, seed: int, **kwargs):
    dict_args = vars(args)
    set_seeds(args.base_seed+seed)

    dataset, data = getattr(datasets, args.dataset)()
        
    if args.mode == 'ind':
        data, unobs_test_data = inductive_split(data, args.ratio)
        unobs_test_loader = DataLoader([unobs_test_data], batch_size=1, shuffle=False, pin_memory=True)

    train_loader = DataLoader([data], batch_size=1, shuffle=False, pin_memory=True)
    n_nodes = data.num_nodes
    n_features = dataset.num_node_features
    n_classes = dataset.num_classes

    teacher_models = load_teachers(**dict_args, n_features=n_features, n_classes=n_classes, seed=seed) # todo: edge_indexいらないかも
    
    model = create_model(in_channels=n_features, out_channels=n_classes,
                         teacher_models=teacher_models, **dict_args)
    early_stopping = EarlyStopping(monitor='val_acc', patience=50, mode='max')
    loggers = get_loggers(args)

    if args.output_only_results:
        trainer = pl.Trainer(accelerator='gpu', gpus=[int(args.device)], enable_model_summary=False,
                        enable_checkpointing=False, logger=False,
                        max_epochs=2000, callbacks=[early_stopping], progress_bar_refresh_rate=0)
    else:
        trainer = pl.Trainer(accelerator='gpu', gpus=[int(args.device)], enable_model_summary=False,
                        enable_checkpointing=False, logger=False,
                        max_epochs=2000, callbacks=[early_stopping])

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=train_loader)

    if args.mode == 'tran':
        
        if args.output_only_results:
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                result = trainer.test(model, dataloaders=train_loader)
                
        else:
            result = trainer.test(model, dataloaders=train_loader)
        
    elif args.mode == 'ind':
        
        result = []
        
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            
            if args.ratio != 0.0:
                obs_result = trainer.test(model, dataloaders=train_loader)
            else:
                obs_result = [{'test_acc': 0.0}]
                
            unobs_result = trainer.test(model, dataloaders=unobs_test_loader)
            
            result.append({'obs_test_acc': obs_result[0]['test_acc']})
            result.append({'unobs_test_acc': unobs_result[0]['test_acc']})
            result.append({'test_acc': args.ratio * obs_result[0]['test_acc'] + (1-args.ratio) * unobs_result[0]['test_acc']})
            
    if args.save:
        trainer.save_checkpoint(Path(args.save_dir)/f'{args.dataset}_{args.method}_{args.teacher_models_name}_{args.teacher_hidden_channels}_{args.teacher_layers}_{seed}.ckpt')
    
    return result


def main(args):
    results = []
    for i in range(0, args.n_runs):
        
        result = train(args, seed=i)
        results.append(ChainMap(*result))
        
        if not args.output_only_results:
            print('-'*100)
        
    print(args)
    
    if args.mode == 'tran':
        df = pd.DataFrame(results)
    elif args.mode == 'ind':
        df = pd.DataFrame(results)[['obs_test_acc', 'unobs_test_acc', 'test_acc']]
    
    print(df.describe().apply(lambda s: s.apply('{0:.3f}'.format)))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--teacher_models_name', type=str, nargs='+', default=['GCN', 'GCN'])
    parser.add_argument('--teacher_hidden_channels', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--teacher_layers', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--method', type=str, default='attn')
    parser.add_argument('--n_runs', type=int, default=10)
    parser.add_argument('--config_name', type=str, default='./train_conf.yaml')
    parser.add_argument('--device', type=str, default='-1') 
    parser.add_argument('--save', type=str, default=False) 
    parser.add_argument('--mode', type=str, default='tran') 
    parser.add_argument('--ratio', type=float, default=0.8) 
    parser.add_argument('--base_seed', type=int, default=77)
    parser.add_argument('--output_only_results', type=bool, default=False)
    
    temp_args, _ = parser.parse_known_args()
    if temp_args.method == 'attn':
        parser = LitDistilledModel.add_model_specific_args(parser)
    elif temp_args.method == 'glnn':
        parser = LitGLNN.add_model_specific_args(parser)
    else:
        raise NameError('Invalid method')
    config = get_training_config(temp_args.config_name, 'MLP', temp_args.dataset)
    parser.set_defaults(**config)
    
    args = parser.parse_args()
    args.save_dir = f'./lightning_logs/{args.dataset}'
    args.exp_group_name = str(int(time()))

    if not args.output_only_results:
        print(args)
    
    main(args)
