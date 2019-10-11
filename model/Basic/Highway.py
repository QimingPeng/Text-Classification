import torch.nn as nn
import torch.nn.functional as F
import torch


class Highway(nn.Module):

    def __init__(self, dim=256):
        super(Highway, self).__init__()
        self.linear = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)


    def forward(self, feature):
        gate = torch.sigmoid(self.gate(feature))
        nonlinear = F.relu(self.linear(feature))
        output = gate * nonlinear + (1 - gate) * feature
        return output
