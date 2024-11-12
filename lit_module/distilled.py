import torch
import torch.nn.functional as F

from distillers.kd import MultiKLDistiller

from models import MLP

from .training_module import LitBaseGNN


class LitDistilledModel(LitBaseGNN):
    def __init__(self, in_channels, out_channels, teacher_models, **kwargs):
        super(LitDistilledModel, self).__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
        self.save_hyperparameters(ignore=['teacher_models'])
        self.teacher_models = teacher_models
  
        self.model = MLP([in_channels, self.hparams.hidden_channels, out_channels],
                            dropout=self.hparams.dropout, batch_norm=False)

        self.kd = MultiKLDistiller(T=self.hparams.temperature)
            
        self.teahcer_outputs = None
            

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('LitDistilledModel')
        
        parser.add_argument('--hidden_channels', type=int, default=64)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.8)
        parser.add_argument('--lr', type=float, default=5e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--lamb', type=float, default=0.0)
        
        parser.add_argument('--agg', type=str, default='attn')
        parser.add_argument('--att_scaling', type=bool, default=False)
        parser.add_argument('--noise', type=float, default=0.0)
        parser.add_argument('--temp_att', type=float, default=1.0) # temperature for attention
        parser.add_argument('--smooth_loss_weight', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=128)
        
        # For Attention Mechanism
        parser.add_argument('--attn_hidden_channels', type=int, default=64)
        parser.add_argument('--attn_max_epoch', type=int, default=5)
        parser.add_argument('--attn_patience', type=int, default=1000)
        parser.add_argument('--attn_lr', type=float, default=0.001)
        parser.add_argument('--attn_weight_decay', type=float, default=0.0)
        
        return parent_parser

    def forward(self, x, edge_index):
        return self.model(x, edge_index)
    
    def training_step(self, data, batch_idx):
        
        x = data.x
        
        if self.hparams.noise > 0:
            noise = self.hparams.noise * torch.abs(F.normalize(torch.randn_like(data.x))) * torch.sign(data.x)
            x += noise
            
        out = self(x, data.edge_index)
        loss1 = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        
        # use cache to avoid computing teacher_outputs every time
        if (self.teahcer_outputs is None) or (self.hparams.noise > 0.0):
            teahcer_outputs = self._compute_teacher_output(x, data.edge_index)
            self.teahcer_outputs = teahcer_outputs
        else:
            teahcer_outputs = self.teahcer_outputs

        
        if self.hparams.lamb == 1.0:
            loss2 = torch.tensor(0.0).to(out.device)
            
        else :
            loss2 = self.kd(out, teahcer_outputs, agg=self.hparams.agg, att_scaling=self.hparams.att_scaling, batch_size=self.hparams.batch_size, \
                temp_att=self.hparams.temp_att, data=data, smooth_loss_weight=self.hparams.smooth_loss_weight, \
                attn_hidden_channels=self.hparams.attn_hidden_channels, attn_max_epoch=self.hparams.attn_max_epoch, attn_patience=self.hparams.attn_patience,\
                attn_lr=self.hparams.attn_lr, attn_weight_decay=self.hparams.attn_weight_decay)
        
        loss = self.hparams.lamb * loss1 + (1 - self.hparams.lamb) * loss2
    
        self.train_acc(out.softmax(dim=-1)[data.train_mask], data.y[data.train_mask])
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def _compute_teacher_output(self, x, edge_index):
        teahcer_outputs = []
        for teacher in self.teacher_models:
            teacher.eval()    
            teahcer_outputs.append(teacher(x, edge_index))
        return teahcer_outputs

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
