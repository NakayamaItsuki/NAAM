import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchmetrics.functional import accuracy

from models import GCN


class LitBaseGNN(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 64, num_layers: int = 2, dropout: float = 0.8, **kwargs):
        super().__init__()
        # self.save_hyperparameters()
        self.model = GCN(in_channels, hidden_channels=hidden_channels,
                           out_channels=out_channels, num_layers=num_layers, dropout=dropout)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitGNN")
        parser.add_argument("--hidden_channels", type=int, default=64)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.8)

        return parent_parser

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, data, batch_idx):
        out = self(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        self.train_acc(out.softmax(dim=-1)[data.train_mask], data.y[data.train_mask])
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx):
        pred = self(data.x, data.edge_index).argmax(dim=-1)
        self.val_acc(pred[data.val_mask], data.y[data.val_mask])
        self.log('val_acc', accuracy(pred[data.val_mask], data.y[data.val_mask]),
                 prog_bar=True, on_step=False, on_epoch=True)
        
    def test_step(self, data, batch_idx, dataloader_id=0):
        pred = self(data.x, data.edge_index).argmax(dim=-1)
        self.test_acc(pred[data.test_mask], data.y[data.test_mask])
        self.log('test_acc', accuracy(pred[data.test_mask], data.y[data.test_mask]),
                 prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-3)
