import torch.nn as nn
import torch.nn.functional as F
import torch

from model.Basic.BiLSTM import BiLSTM

class TextRCNN_Config(object):
    def __init__(self, pretrained_embedding, max_seq_len, num_label):
        self.pretrained_embedding = pretrained_embedding
        self.vocab_size = pretrained_embedding.size(0)
        self.embed_dim = pretrained_embedding.size(1)
        self.max_seq_len = max_seq_len
        self.hidden_size = 128
        self.num_label = num_label
        self.num_layers = 2
        self.dropout = 0.3


class TextRCNN(nn.Module):

    def __init__(self, config):
        super(TextRCNN, self).__init__()
        if config.pretrained_embedding is None:
            self.embedding_layer = nn.Embedding(config.vocab_size, config.embed_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                config.pretrained_embedding, freeze=False)
        self.BiLSTM = BiLSTM(config.embed_dim, config.hidden_size, config.num_layers, rnn_type="LSTM")
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed_dim, config.hidden_size * 2)
        self.maxpool = nn.MaxPool1d(config.max_seq_len)
        self.classfier = nn.Linear(config.hidden_size * 2, config.num_label)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        sentences, sentence_lengths = x
        # text: [sent len, batch size]
        embeds = self.embedding(sentences)   # [batch_size, max_seq_len, embed_dim]

        lstm_output, _ = self.BiLSTM(embeds, sentence_lengths)  # [batch_size, max_seq_len, hidden_size*2]

        fc_input = torch.cat((lstm_output, embeds), dim=-1)     # [batch_size, max_seq_len, hidden_size*2 + embed_dim]
        fc_output = torch.tanh(self.fc(fc_input))   # [batch_size, max_seq_len, hidden_size*2]

        maxpool_input = fc_output.permute(0, 2, 1)  # [batch_size, hidden_size*2, max_seq_len]
        maxpool_output = self.maxpool(maxpool_input).squeeze()   # [batch_size, hidden_size*2]

        cls_input = self.dropout(maxpool_output)
        logits = self.classfier(cls_input)
        return logits