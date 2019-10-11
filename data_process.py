import csv
import os
import random
import bcolz
import torch
import logging
import pyhanlp
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split



random.seed (2019)

def load_csv(data_path):
    """
        载入csv文件
    """
    with open(data_path, "r", encoding="utf-8") as f:
        reader=csv.reader(f)
        data = []
        for line in reader:
            data.append(line)
        return data[1:]

def write_csv(data_list, data_path):
    """
        将结果写入csv文件
    """
    with open(data_path, 'w', encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        for row in data_list:
            writer.writerow(row)


def load_embeddings(embed_path, extra_token_dict):
    """
        从 bcolz 加载 词/字 向量，并构造相应的词典
        Args:
            embed_path (str): 解压后的 bcolz rootdir（如 zh.64），
                                里面包含 2 个子目录 embeddings 和 words，
                                分别存储 嵌入向量 和 词（字）典
        Returns:
            word2id, id2word (dict): 构造词id字典
            embeddings (torch.Tensor): 嵌入矩阵，每 1 行为 1 个 词向量/字向量，
                                       其行号对应在 word2id 中的value
    """
    embed_path = embed_path.rstrip('/')
    # 词（字）典列表（bcolz carray具有和 numpy array 类似的接口）
    words = bcolz.carray(rootdir='%s/words'%embed_path, mode='r')
    embeddings = bcolz.carray(rootdir='%s/embeddings'%embed_path, mode='r')

    words = list(words)
    embeddings = torch.tensor(embeddings, dtype=torch.float)

    embed_dim = embeddings.size()[1]

    unk_token = extra_token_dict["unk_token"]
    pad_token = extra_token_dict["pad_token"]

    if unk_token not in words:
        words.append(unk_token)
        unk_tensor = torch.randn(1, embed_dim)
        embeddings = torch.cat((embeddings, unk_tensor), dim=0)

    if pad_token not in words:
        words.append(pad_token)
        pad_tensor = torch.zeros(1, embed_dim)
        embeddings = torch.cat((embeddings, pad_tensor), dim=0)

    word2id = {}
    for word in words:
        word2id[word] = len(word2id)
    id2word = dict(zip(word2id.values(),  word2id.keys()))
    return word2id, id2word, embeddings



def read_items(data):
    input_sentences = []
    labels = []
    logging.info("words segmentattion...")
    i = 0
    for item in tqdm(data):
        i += 1
        temp_list = pyhanlp.HanLP.segment(item[1])
        words = [str(i.word) for i in temp_list]
        # temp_list = jieba.cut(item[0]) 
        # print(" ".join(temp_list))
        if " ".join(words) == 0:
            continue
        input_sentences.append(" ".join(words))
        labels.append(int(item[0]))
    return input_sentences, labels

def data_preprocess(data_path):
    raw_data = load_csv(data_path)
    # moods_list = ['喜悦', '愤怒', '厌恶', '低落']
    # count_list = [0] * 4
    logging.info('样本总数：%d' % len(raw_data))
    # for item in raw_data: 
    #     count_list[int(item[0])] += 1
    # for i in range(4):
        # logging.info('微博数目（{}）：{}'.format(moods_list[i],  count_list[i]))
    
    input_sentences, labels = read_items(raw_data)
    max_len = max([len(sentence.split()) for sentence in input_sentences])

    logging.info("分词后最大文本长度：{}".format(max_len))
    train_sentences, test_sentences, train_labels, test_labels = train_test_split(input_sentences, labels, test_size=0.1, stratify=labels)
    return train_sentences, test_sentences, train_labels, test_labels


def load_data(raw_path, embed_path, train_path, test_path, max_seq_len, extra_token_dict):

    word2id, id2word, embeddings = load_embeddings(embed_path, extra_token_dict)

    if not os.path.exists(train_path) and not os.path.exists(train_path):
        logging.info("Data preprocess...")
        train_dev_sentences, test_sentences, train_dev_labels, test_labels = data_preprocess(raw_path)
        train = [("review", "label")] + list(zip(train_dev_sentences, train_dev_labels))
        write_csv(train, train_path)
        test = [("review", "label")] + list(zip(test_sentences, test_labels))
        write_csv(test, test_path)

    logging.info("Load data from train test data...")
    train_dev_data = load_csv(train_path)
    train_dev_sentences = []
    train_dev_labels = []
    test_sentences = []
    test_labels = []

    for item in train_dev_data:
        train_dev_sentences.append(item[0].split()[:max_seq_len])
        train_dev_labels.append(int(item[1]))
    test_data = load_csv(test_path)
    for item in test_data:
        test_sentences.append(item[0].split()[:max_seq_len])
        test_labels.append(int(item[1]))
    
    return train_dev_sentences, test_sentences, train_dev_labels, test_labels, word2id, id2word, embeddings
