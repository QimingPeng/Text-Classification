import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch

class Attention(nn.Module):
    def __init__(self, hidden_size, attn_type="Soft_Attention"):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.to_score = nn.Linear(hidden_size, 1, bias=False)  # [hidden_size]

    def init_parameters(self):
        """
            对权重进行初始化
        """
        nn.init.uniform_(self.context_weight, -0.1, 0.1)

    def forward(self, seqs_input, seqs_len):
        batch_size, max_seq_len, _ = seqs_input.size()
        mask = self.creat_attn_mask(batch_size, max_seq_len, seqs_len)
        fc_out = torch.tanh(self.mlp(seqs_input))      # [batch_size, max_seqs_len, hidden_size]
        scores = self.to_score(fc_out)  # [batch_size, max_seqs_len, 1]
        score_masked = scores.masked_fill(mask == 0, -np.inf)

        score = F.softmax(score_masked, dim=1)       # [batch_size, max_seqs_len, 1]

        attn_output = torch.sum(seqs_input * score, dim=1)    # 句子表示 [batch_size, hidden_size]
        # attn_output = torch.tanh(attn_output)

        return attn_output

    def creat_attn_mask(self, batch_size, max_seq_len, seqs_len):
        seqs_indices = torch.arange(0, max_seq_len).unsqueeze(0).type_as(seqs_len)
        seqs_indices = seqs_indices.expand(batch_size, max_seq_len)
        seqs_len = seqs_len.unsqueeze(dim=1).expand(batch_size, max_seq_len)
        # returns [batch_size, max_seq_len]
        mask = (seqs_indices < seqs_len).int().detach().unsqueeze(-1) # 不参与训练
        return mask