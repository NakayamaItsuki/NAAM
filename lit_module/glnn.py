import torch
import torch.nn.functional as F

from distillers.kd import MultiKLDistiller
from models import MLP

from .training_module import LitBaseGNN


class LitGLNN(LitBaseGNN):
    def __init__(self, in_channels, out_channels, teacher_models, **kwargs):
        super(LitGLNN, self).__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.save_hyperparameters(ignore=['teacher_models'])
        self.teacher_models = teacher_models
        
        self.model = MLP([in_channels, self.hparams.hidden_channels, out_channels],
                            dropout=self.hparams.dropout, batch_norm=False)
        
        self.kd = MultiKLDistiller(T=self.hparams.temperature)
        
        self.teahcer_output = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('LitGLNN')
        parser.add_argument('--hidden_channels', type=int, default=64)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--lr', type=float, default=5e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--dropout', type=float, default=0.8)
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--lamb', type=float, default=0.0)

        return parent_parser

    def forward(self, x, edge_index):
        return self.model(x, edge_index)

    def training_step(self, data, batch_idx):
        out = self(data.x, data.edge_index)
        loss1 = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        if self.teahcer_output is None:
            self.teacher_models[0].eval()
            self.teahcer_output = self.teacher_models[0](data.x, data.edge_index)
        
        if self.hparams.lamb == 1.0:
            loss2 = 0
        else:
            loss2 = self.kd(out, [self.teahcer_output])
        
        loss = self.hparams.lamb * loss1 + (1 - self.hparams.lamb) * loss2
        self.train_acc(out.softmax(dim=-1)[data.train_mask], data.y[data.train_mask])
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
