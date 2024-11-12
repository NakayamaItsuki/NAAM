import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_mechanism import AttentionMechanism

class MultiKLDistiller(nn.Module):
    def __init__(self, T=1):
        super(MultiKLDistiller, self).__init__()
        self.T = T
        
        self.attention_mechanism = None

    def forward(self, s_output, t_output, agg='attn', att_scaling=False,  batch_size=None, temp_att=1.0, data=None, smooth_loss_weight=0.0, \
                attn_hidden_channels=64, attn_max_epoch=5, attn_patience=1000, attn_lr=0.001, attn_weight_decay=0.0):
        
        if agg == 'separated':
            s_p = F.log_softmax(s_output / self.T, dim=-1)
            loss_list = []
            for t_p in t_output:
                t_p = F.log_softmax(t_p / self.T, dim=-1)
                loss = F.kl_div(s_p, t_p, reduction='batchmean', log_target=True) * (self.T * self.T)
                loss_list.append(loss)
            return sum(loss_list)

        # For glnn
        if len(t_output) == 1:
            t_p = t_output[0]

        else:
            t_p = self._compute_teacher_y(s_output, t_output, agg, att_scaling, batch_size, temp_att, data, attn_hidden_channels, attn_max_epoch, \
                                            attn_patience, attn_lr, attn_weight_decay, smooth_loss_weight)
        s_output_ = F.log_softmax(s_output / self.T, dim=-1)
        if agg != 'mean':
            t_p = F.log_softmax(t_p / self.T, dim=-1)

        loss = F.kl_div(s_output_, t_p, reduction='batchmean', log_target=True) * (self.T * self.T)

        return loss

    @torch.no_grad()
    def _compute_teacher_y(self, s_output, t_output, agg, att_scaling, batch_size=None, temp_att=1.0, data=None, attn_hidden_channels=64, attn_max_epoch=5, attn_patience=1000, attn_lr=0.001, attn_weight_decay=0.0, smooth_loss_weight=0.0) -> torch.Tensor:
        
        t_output = torch.stack(t_output, dim=0)
        
        if agg == 'sum':
            y_soft = torch.sum(t_output, dim=0)
            
        elif agg == 'mean':
            y_soft = torch.mean(F.log_softmax(t_output/self.T, dim=-1), dim=0)
            
        elif agg == 'attn':
            if batch_size is not None:
                alphas = self.batch_at(s_output, t_output, batch_size)
            else:
                alphas = self.at(s_output, t_output)
            
            y_soft = torch.einsum('i,ijk->ijk', [alphas, t_output]).sum(dim=0)
        
        elif agg == 'learnable_attn':
            if self.attention_mechanism == None:
                self.attention_mechanism = AttentionMechanism(agg, att_scaling, temp_att, data, max_epochs=attn_max_epoch, in_channels=s_output.size(-1), hidden_channels=attn_hidden_channels, \
                    lr=attn_lr, weight_decay=attn_weight_decay, patience=attn_patience, smooth_loss_weight=smooth_loss_weight)

            with torch.enable_grad():
                y_soft = self.attention_mechanism(s_output, t_output)

        else:
            raise ValueError(f'Invalid aggregate method. Expect [sum | mean], but got {agg}')
        
        return y_soft
    
    
    @torch.no_grad()
    def at(self, s_output, t_output):
        s_output = F.normalize(s_output, dim=1)
        alphas = []
        for t in t_output:
            alpha = torch.mm(s_output, F.normalize(t, dim=1).t()).mean()
            alphas.append(alpha)
        alphas = torch.tensor(alphas, device=s_output.device)
        alphas /= alphas.sum()
        alphas = F.softmax(alphas, dim=0)
        return alphas

    @torch.no_grad()
    def batch_at(self, s_output, t_output, batch_size):
        s_output = F.normalize(s_output, dim=1)
        alphas = []
        for t in t_output:
            t = F.normalize(t, dim=1)
            num_nodes = t.size(0)
            num_batches = (num_nodes - 1) // batch_size + 1
            indices = torch.arange(0, num_nodes).to(t.device)
            batch_alpha = []
            for i in range(num_batches):
                mask = indices[i * batch_size:(i + 1) * batch_size]
                batch_alpha.append(torch.mm(s_output[mask], t.t()).mean())
            alpha = torch.tensor(batch_alpha).mean()
            alphas.append(alpha)
        alphas = torch.tensor(alphas, device=s_output.device)
        alphas /= alphas.sum()
        alphas = F.softmax(alphas, dim=0)
        return alphas

    @torch.no_grad()
    def at2(self, s_output, t_output):
        s_output = F.log_softmax(s_output / self.T, dim=-1)

        alphas = []
        for t in t_output:
            alpha = torch.mm(s_output, F.log_softmax(t / self.T, dim=-1).t()).mean()
            alphas.append(alpha)
        alphas = torch.tensor(alphas, device=s_output.device)
        alphas = F.softmax(alphas, dim=0)
        return alphas

    @torch.no_grad()
    def at3(self, s_output, t_output):
        alphas = []
        s = s_output @ s_output.t()

        for t in t_output:
            t_m = t @ t.t()
            alpha = (s-t_m).pow(2).mean()
            alphas.append(alpha)
        alphas = torch.tensor(alphas, device=s_output.device)
        alphas /= alphas.sum()
        alphas = F.softmax(alphas, dim=0)
        return alphas
    
    
    
