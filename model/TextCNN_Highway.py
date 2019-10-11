import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Basic.Conv1d import Conv1d
from model.Basic.Highway import Highway


class TextCNN_Highway_Config(object):
    def __init__(self, pretrained_embedding, num_label):
        self.pretrained_embedding = pretrained_embedding
        self.vocab_size = pretrained_embedding.size(0)
        self.embed_dim = pretrained_embedding.size(1)
        # self.hidden_size = max_seq_len
        self.num_label = num_label
        self.dropout = 0.3
        self.n_kernel = 128

        self.kernel_size = [1, 2, 3, 4, 5]


class TextCNN_Highway(nn.Module):
    def __init__(self, config):
        super(TextCNN_withHighway, self).__init__()
        if config.pretrained_embedding is None:
            self.embedding_layer = nn.Embedding(config.vocab_size, config.embed_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                config.pretrained_embedding, freeze=False)
        self.convs = Conv1d(config.embed_dim, config.n_kernel, config.kernel_size)
        self.highway = Highway(config.n_kernel * len(config.kernel_size))
        self.classfier = nn.Linear(len(config.kernel_size) * config.n_kernel, config.num_label)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        text, _ = x
        embedded = self.embedding(text)     # [batch_size, max_len, embed_dim]
        
        conved = self.convs(embedded)       # [batch_size, n_filters, max_len - kernel_size[n] - 1]

        pooled = [F.max_pool1d(conv, conv.size(-1)).squeeze(2)      #[len(kernel_size), batch_size, n_kernel]
                  for conv in conved]

        pooled_out = torch.cat(pooled, dim=1)      # [batch_size, n_kernel * len(kernel_size)]

        highway_output = self.highway(pooled_out)

        cls_input = self.dropout(highway_output)
        logits = self.classfier(cls_input)

        return logits