from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from tqdm import trange

from utils import EarlyStopping


class Trainer():
    def __init__(self, model, optimizer, data, device, epochs=2000, patience=50, ratio=0.2, saved_path='model_best.pt'):
        self.model = model
        self.optimizer = optimizer
        self.dataset = data
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.saved_path = Path(saved_path)
        self.saved_path.parent.mkdir(parents=True, exist_ok=True)
        self.ratio = ratio

    def train(self):
        early_stopping = EarlyStopping(patience=self.patience, path=self.saved_path)
        with trange(self.epochs) as t:
            for epoch in t:
                loss, val_loss = self._train()
                train_acc, val_acc, test_acc = self.test(self.ratio)
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    break
                t.set_postfix(val_acc=val_acc, test_acc=test_acc)
        self.model.load_state_dict(torch.load(self.saved_path))

    def _train(self):
        self.model.train()
        self.optimizer.zero_grad()
        if hasattr(self.dataset, 'adj_t'):
            out = self.model(self.dataset.x, self.dataset.adj_t)
        else:
            out = self.model(self.dataset.x, self.dataset.edge_index)

        loss = F.cross_entropy(out[self.dataset.train_mask], self.dataset.y[self.dataset.train_mask])

        with torch.no_grad():
            val_loss = F.cross_entropy(out[self.dataset.val_mask], self.dataset.y[self.dataset.val_mask])

        loss.backward()
        self.optimizer.step()
        return float(loss), val_loss

    @torch.no_grad()
    def test(self, ratio=None):
        self.model.eval()
        pred = self.model(self.dataset.x, self.dataset.edge_index).argmax(dim=-1)

        accs = []
        
        if ratio != 0.0:
            for _, mask in self.dataset('train_mask', 'val_mask', 'test_mask'):
                accs.append(accuracy(pred[mask], self.dataset.y[mask]).item())
        else:
            for _, mask in self.dataset('train_mask', 'val_mask'):
                accs.append(accuracy(pred[mask], self.dataset.y[mask]).item())
            accs.append(0.0)
        return accs
