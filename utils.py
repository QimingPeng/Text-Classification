import pickle
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch
from sklearn.metrics import f1_score


class WarmupLinearSchedule(LambdaLR):
    """ 
        学习率调整策略
        Args:
            optimizer : 需要调整的优化器
            warmup_steps : 从开始到warmup_steps步学习率线性上升
            t_total : 训练的总步数，从warmup_steps到训练结束学习率线性下降
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

def macro_f1(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    return f1_score(labels, outputs, labels=[0, 1],average='macro')

def convert2feature(input_sentences, labels, word2id, max_seq_len):
        pad_id = word2id.get('<PAD>')
        unk_id = word2id.get('<UNK>')
        sentence_len = []
        data_tensor = torch.ones(len(input_sentences), max_seq_len).long() * pad_id
        for i, sentence in enumerate(input_sentences):
            sentence_len.append(len(sentence))
            for j, word in enumerate(sentence[:max_seq_len]):
                data_tensor[i][j] = word2id.get(word, unk_id)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        sentence_len_tensor = torch.tensor(sentence_len, dtype=torch.long)
        return data_tensor, sentence_len_tensor, labels_tensor
