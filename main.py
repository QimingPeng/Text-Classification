import argparse
import csv
import os
import random
import torch
import logging
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from itertools import cycle
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from data_process import load_data
from utils import convert2feature, macro_f1
from model.TextRNN import TextRNN, TextRNN_Config
from model.TextCNN import TextCNN, TextCNN_Config
from model.TextRCNN import TextRCNN, TextRCNN_Config
from model.TextCNN_Highway import TextCNN_Highway, TextCNN_Highway_Config
from model.TextRNN_Attention import TextRNN_Attention, TextRNN_Attention_Config


from train_test import train_eval, test


def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-raw_path', default="./data/waimai_10k.csv")
    parser.add_argument('-train_path', default="./data/train.csv")
    parser.add_argument('-test_path', default="./data/test.csv")
    parser.add_argument('-embedding_path',  help='Embedding for words', default='./data/zh.256')
    parser.add_argument('-num_label', default=2, type=int)

    parser.add_argument('-do_train', type=bool, default=True)
    parser.add_argument('-do_cv', type=bool, default=True)
    parser.add_argument('-do_test', type=bool, default=True)

    parser.add_argument('-model_name', default="TextRNN_Attention", type=str)

    parser.add_argument('-max_seq_len', default=300, type=int)
    parser.add_argument('-seed', default=2019, type=int)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-dev_batch_size', default=32, type=int)
    parser.add_argument('-train_steps', default=7000, type=int)
    parser.add_argument('-check_step', default=100, type=int)
    parser.add_argument('-eval_step', default=500, type=int)

    parser.add_argument('-lr', default=5e-4, type=float)
    parser.add_argument('-warmup_steps', default=0, type=int)

    parser.add_argument('-ckpt_path', default="./ckpts/")
    return parser.parse_args()

def set_seed(seed, n_gpu):
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main(args):
    # 可选择输出日志到文件
    # logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
    #                     filename=args.model_name + '.log',
    #                     filemode='w',
    #                     level=logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                        level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    set_seed(args.seed, n_gpu)

    logging.info("The active device is: {}, gpu_num: {}".format(device, n_gpu))

    args.ckpt_path = os.path.join(args.ckpt_path, args.model_name)
    try:
        os.makedirs(args.ckpt_path)
    except:
        pass
    
    extra_token_dict = {"unk_token": "<UNK>", "pad_token": "<PAD>"}

    train_dev_sentences, test_sentences, train_dev_labels, test_labels, word2id, id2word, embeddings = \
        load_data(args.raw_path, args.embedding_path, args.train_path, args.test_path, args.max_seq_len, extra_token_dict)

    f1_list = [0] * 5
    if args.do_train:
        kf = StratifiedKFold(n_splits=5, shuffle=True).split(train_dev_sentences, train_dev_labels)
        for cv_i, (train_index, dev_index) in enumerate(kf):
            logging.info("******************Train CV_{}******************".format(cv_i))
            # 准备模型
            if args.model_name == "TextRNN":
                config = TextRNN_Config(embeddings, args.num_label)
                model = TextRNN(config)
            if args.model_name == "TextCNN":
                config = TextCNN_Config(embeddings, args.num_label)
                model = TextCNN(config)
            if args.model_name == "TextRCNN":
                config = TextRCNN_Config(embeddings, args.max_seq_len, args.num_label)
                model = TextRCNN(config)
            if args.model_name == "TextCNN_Highway":
                config = TextCNN_Highway_Config(embeddings, args.num_label)
                model = TextCNN_Highway(config)
            if args.model_name == "TextRNN_Attention":
                config = TextRNN_Attention_Config(embeddings, args.num_label)
                model = TextRNN_Attention(config)
            logging.info("Already load the model: {},".format(args.model_name))
            model.to(device)

            train_sentences = [train_dev_sentences[i] for i in train_index]
            train_labels = [train_dev_labels[i] for i in train_index]
            dev_sentences = [train_dev_sentences[i] for i in dev_index]
            dev_labels = [train_dev_labels[i] for i in dev_index]

            logging.info("Prepare dataloader...")
            train_tensor, train_sent_len, train_labels_tensor = convert2feature(train_sentences, train_labels, word2id, args.max_seq_len)
            train_data = TensorDataset(train_tensor, train_sent_len, train_labels_tensor)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
            train_dataloader=cycle(train_dataloader) 

            dev_tensor, dev_sent_len, dev_labels_tensor = convert2feature(dev_sentences, dev_labels, word2id, args.max_seq_len)            
            dev_data = TensorDataset(dev_tensor, dev_sent_len, dev_labels_tensor)
            dev_sampler = SequentialSampler(dev_data)
            dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=args.dev_batch_size)

            logging.info("Begin to train...")
            f1_list[cv_i] = train_eval(train_dataloader, dev_dataloader, model, args.ckpt_path, 
                                        args.train_steps, args.check_step, args.eval_step, args.lr, args.warmup_steps, cv_i)
            if not args.do_cv:
                break
        if args.do_cv:
            cv_f1 = np.mean(np.array(f1_list))
            logging.info("CV F1_list: {}, Mean_F1: {:.4f}".format(f1_list, cv_f1))

    if args.do_test:
        logging.info("******************Test******************")
        logging.info("Begin to test {}...".format(args.model_name))
        test_tensor, test_sent_len, test_labels_tensor = convert2feature(test_sentences, test_labels, word2id, args.max_seq_len)
        test_data = TensorDataset(test_tensor, test_sent_len, test_labels_tensor)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.dev_batch_size)

        final_results = np.zeros((len(test_labels), args.num_label))
        test_labels = test_labels_tensor.to('cpu').numpy()
        for cv_i in range(5):
            ckpt_path = os.path.join(args.ckpt_path, "pytorch_model_{}.pkl".format(cv_i))
            if args.model_name == "TextRNN":
                config = TextRNN_Config(embeddings, args.num_label)
                model = TextRNN(config)
                model.load_state_dict(torch.load(ckpt_path))
                model.to(device)
            if args.model_name == "TextCNN":
                config = TextCNN_Config(embeddings, args.num_label)
                model = TextCNN(config)
                model.load_state_dict(torch.load(ckpt_path))
                model.to(device)
            if args.model_name == "TextRCNN":
                config = TextRCNN_Config(embeddings, args.max_len, args.num_label)
                model = TextRCNN(config)
                model.load_state_dict(torch.load(ckpt_path))
                model.to(device)
            if args.model_name == "TextRNN_Attention":
                config = TextRNN_Attention_Config(embeddings, args.num_label)
                model = TextRNN_Attention(config)
                model.load_state_dict(torch.load(ckpt_path))
                model.to(device)
            output_labels, test_f1_score = test(test_dataloader, model, device, args.dev_batch_size)
            final_results = final_results + output_labels
            logging.info("The cv_{} result of {} on test data: F1: {:.4f}".format(cv_i, args.model_name, test_f1_score))
            if not args.do_cv:
                break

        test_f1_score = round(macro_f1(final_results, test_labels), 4)
        logging.info("The final result of {} on test data: F1: {:.4f}".format(args.model_name, test_f1_score))

if __name__ == "__main__":
    args = init_argparse()
    main(args)

    

