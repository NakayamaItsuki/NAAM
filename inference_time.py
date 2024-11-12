from argparse import ArgumentParser

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

import datasets
from models import GAT, GCN, SGC, MLP, GCN2, APPNP
from utils import get_training_config

cudnn.benchmark = True


def measure_inference_time(model, repetitions: int, *args, **kwargs):
    model.eval()
    print('warm up ...')
    with torch.no_grad():
        for _ in range(100):
            model(*args, **kwargs)

    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    print('testing ...')
    with torch.no_grad():
        for rep in tqdm.tqdm(range(repetitions)):
            starter.record()
            model(*args, **kwargs)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # ms
            timings[rep] = curr_time

    avg = timings.sum()/repetitions
    print(f'avg={avg}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon_computers')
    parser.add_argument('--model_name', type=str, default='MLP')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)

    parser.add_argument('--n_repetitions', type=int, default=1000)
    parser.add_argument('--device', type=str, default='0')
    args = parser.parse_args()

    device = 'cuda:'+args.device
    dataset, data = getattr(datasets, args.dataset)()
    data.to(device)
    n_features = dataset.num_node_features
    n_classes = dataset.num_classes

    model_name = args.model_name
    hidden_channels = args.hidden_channels
    num_layers = args.num_layers
    
    dataset = args.dataset

    if model_name == 'GCN':
        model=GCN(n_features, hidden_channels=hidden_channels,
                        out_channels=n_classes, num_layers=num_layers)
    elif model_name == 'GAT':
        model=GAT(n_features, hidden_channels=hidden_channels,
                        out_channels=n_classes, num_layers=num_layers)
    elif model_name == 'SGC':
        model=SGC(in_channels=n_features, out_channels=n_classes, num_layers=num_layers, cached=False)
    elif model_name == 'GCN2':
        model=GCN2(in_channels=n_features, hidden_channels=hidden_channels,
                        out_channels=n_classes, num_layers=num_layers)
    elif model_name == 'APPNP':
        config = get_training_config('./train_conf.yaml', 'APPNP', dataset)
        model=APPNP(in_channels=n_features, hidden_channels=hidden_channels,
                        out_channels=n_classes, num_layers=num_layers, K=config['K'], alpha=config['alpha'])
    elif model_name == 'MLP':
        model=MLP([n_features]+[hidden_channels]*(args.num_layers-1)+[n_classes])
    else:
        raise NameError('Invalid model type')

    model.to(device)

    measure_inference_time(model, args.n_repetitions, data.x, data.edge_index)
