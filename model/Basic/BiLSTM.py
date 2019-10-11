import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    """
        BiLSTM的实现
        Args:
            feature_dim (int): 特征的维度
            hidden_size (int): lstm隐藏层维度
            num_layers (int): lstm隐藏层的层数
            batch_first (Boolean): 第一个维度是否为batch
            rnn_type (str): 可选择使用LSTM或者GRU
    """
    def __init__(self, feature_dim, hidden_size, num_layers, batch_first=True, rnn_type="LSTM"):
        super(BiLSTM, self).__init__()
        if rnn_type == "LSTM":
            self.bilstm = nn.LSTM(feature_dim, hidden_size, num_layers, 
                                batch_first=batch_first,
                                bidirectional=True)
        else:
            self.bilstm = nn.GRU(feature_dim, hidden_size, num_layers, 
                                batch_first=batch_first,
                                bidirectional=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, seq_embed, lengths):
        seq_embed = self.dropout(seq_embed)
        total_length = seq_embed.size()[1]
        packed = pack_padded_sequence(seq_embed, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, _) = self.bilstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=total_length)
        return output, hidden
