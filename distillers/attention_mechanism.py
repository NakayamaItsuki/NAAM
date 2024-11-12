
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import copy

class AttentionMechanism(nn.Module):
    def __init__(self, agg, att_scaling, temp_att, data, max_epochs, in_channels, hidden_channels, lr, weight_decay, patience, smooth_loss_weight=0.0):
        super(AttentionMechanism, self).__init__()
        
        self.att_scaling = att_scaling
        self.temp_att = temp_att
        self.max_epochs = max_epochs
        self.patience = patience

        self.data = data

        self.w1 = nn.Linear(in_channels, hidden_channels).to(data.x.device)
        self.w2 = nn.Linear(in_channels, hidden_channels).to(data.x.device)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.accuracy = Accuracy().to(data.x.device)

        self.previous_weights = None
        
        self.smooth_loss_weight = smooth_loss_weight
        self.diff_weights_loss = 0.0
        
        
    def forward(self, s_output, t_output):
        
        s_output = s_output.detach()
        t_output = t_output.detach()
        
        best_train_accuracy = 0
        
        patience_cnt = 0
        best_epoch = 0
        
        for epoch in range(self.max_epochs):
            self.train_step(s_output, t_output)
            train_accuracy, val_accuracy, test_accuracy = self.eval_step(s_output, t_output)
            
            if train_accuracy >= best_train_accuracy:
                best_train_accuracy = train_accuracy
                patience_cnt = 0
                best_epoch = epoch
                
                best_w1_state_dict = copy.deepcopy(self.w1.state_dict())
                best_w2_state_dict = copy.deepcopy(self.w2.state_dict())
                
            else:
                patience_cnt += 1
                
            if patience_cnt > self.patience:
                break
            
        self.w1.load_state_dict(best_w1_state_dict)
        self.w2.load_state_dict(best_w2_state_dict)
        
        y_soft = self.compute_attention(s_output, t_output)

        y_soft = y_soft.detach()
        
        return y_soft


    def train_step(self, s_output, t_output):
        self.train()
        self.optimizer.zero_grad()
        
        y_soft = self.compute_attention(s_output, t_output)

        loss = F.cross_entropy(y_soft[self.data.val_mask], self.data.y[self.data.val_mask])
        
        if self.smooth_loss_weight > 0.0:
            loss = loss + self.smooth_loss_weight * self.diff_weights_loss

        loss.backward()
        self.optimizer.step()


    def eval_step(self, s_output, t_output):
        self.eval()
        with torch.no_grad():
            y_soft = self.compute_attention(s_output, t_output)
            
            train_accuracy = self.acc(y_soft, self.data.y, self.data.train_mask)
            val_accuracy = self.acc(y_soft, self.data.y, self.data.val_mask)
            
            test_accuracy = 0.0
            
        return train_accuracy, val_accuracy, test_accuracy


    def compute_attention(self, s_output, t_output):
        t_output_ = self.w1(t_output)
        s_output_ = self.w2(s_output)
        s_output_expanded = s_output_.unsqueeze(0).expand(t_output.size(0), -1, -1)

        kl_divergences = F.kl_div(F.log_softmax(s_output_expanded, dim=-1), F.log_softmax(t_output_, dim=-1), reduction='none', log_target=True).sum(dim=-1)

        kl_divergences = -kl_divergences
        
        if self.att_scaling:
            d_k = s_output.size(-1)
            kl_divergences = kl_divergences / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        weights = F.softmax(kl_divergences / self.temp_att, dim=0)
        
        if self.previous_weights is None:
            initial_weights = 1 /  weights.size(0)
            self.previous_weights = torch.full_like(weights, initial_weights)
            
        self.diff_weights_loss = torch.abs(weights - self.previous_weights).mean()
            
        self.previous_weights = weights

        y_soft = torch.einsum('ij,ijk->jk', [weights, t_output])
    
        return y_soft


    def acc(self, y_soft, y, mask):
        pred = y_soft.argmax(dim=-1)
        return self.accuracy(pred[mask], y[mask])
    
    