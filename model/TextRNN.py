import torch.nn as nn
import torch.nn.functional as F
import torch

from model.Basic.BiLSTM import BiLSTM

class TextRNN_Config(object):
    def __init__(self, pretrained_embedding, num_label):
        self.pretrained_embedding = pretrained_embedding
        self.vocab_size = pretrained_embedding.size(0)
        self.embed_dim = pretrained_embedding.size(1)
        self.hidden_size = 128
        self.num_label = num_label
        self.num_layer = 2
        self.dropout = 0.3


class TextRNN(nn.Module):

    def __init__(self, config):
        super(TextRNN, self).__init__()
        if config.pretrained_embedding is None:
            self.embedding_layer = nn.Embedding(config.vocab_size, config.embed_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                config.pretrained_embedding, freeze=False)
        self.BiLSTM = BiLSTM(config.embed_dim, config.hidden_size, config.num_layer, rnn_type="LSTM")
        self.classfier = nn.Linear(config.hidden_size * 2, config.num_label)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        sentences, sentence_lengths = x
        embeds = self.embedding(sentences)    # [batch_size, max_len, embed_dim]
    
        _, hidden = self.BiLSTM(embeds, sentence_lengths)   # [num_layer*2, batch_size, hidden_size]
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=-1))   # 将最后一层的隐藏层拼接 [batch_size, hidden_size*2]
        logits = self.classfier(hidden)     # [batch_size, num_label]
        return logits