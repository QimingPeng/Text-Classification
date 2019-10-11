import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Basic.BiLSTM import BiLSTM
from model.Basic.Attention import Attention

class TextRNN_Attention_Config(object):
    def __init__(self, pretrained_embedding, num_label):
        self.pretrained_embedding = pretrained_embedding
        self.vocab_size = pretrained_embedding.size(0)
        self.embed_dim = pretrained_embedding.size(1)
        self.hidden_size = 128
        self.num_label = num_label
        self.num_layer = 2
        self.dropout = 0.3


class TextRNN_Attention(nn.Module):
    def __init__(self, config):
        super(TextRNN_Attention, self).__init__()
        if config.pretrained_embedding is None:
            self.embedding_layer = nn.Embedding(config.vocab_size, config.embed_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                config.pretrained_embedding, freeze=False)
        self.BiLSTM = BiLSTM(config.embed_dim, config.hidden_size, config.num_layer, rnn_type="LSTM")
        self.attn = Attention(config.hidden_size*2)
        self.classfier = nn.Linear(config.hidden_size * 2, config.num_label)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        sentences, sentence_lengths = x
        embeds = self.embedding(sentences)    # [batch_size, max_len, embed_dim]
    
        lstm_out, _ = self.BiLSTM(embeds, sentence_lengths)   # [batch_size, max_len, hidden_size*2]

        attn_out = self.attn(lstm_out, sentence_lengths)      # [batch_size, hidden_size*2]
        hidden = self.dropout(attn_out)   # [batch_size, hidden_size*2]
        logits = self.classfier(hidden)     # [batch_size, num_label]

        return logits