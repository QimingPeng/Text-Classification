import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Module):
    """
        定义一个1维卷积层，卷积核大小为in_channels * filter_size，输出的通道数(卷积核个数)为out_channels
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv1d, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=ks)
            for ks in kernel_size
        ])

    def forward(self, feature):
        feature = feature.permute(0, 2, 1)      # [batch_size, feature_dim, seqs_len]
        return [F.relu(conv(feature)) for conv in self.convs]