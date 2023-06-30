import jieba
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from utils import create_dictionaries,text_to_index_array,text_cut_to_same_long,creat_wordvec_tensor


def del_stop_words(text): #分词
    word_ls = jieba.lcut(text)
    if word_ls[0] == '1':
        word_ls = word_ls[5:]
    else: # '-'
        word_ls = word_ls[6:]
    # word_ls = [i for i in word_ls if i not in stopwords]
    return word_ls

with open("raw_data/neg.txt", "r", encoding='UTF-8') as e:  # 加载负面语料
    neg_data1 = e.readlines()

with open("raw_data/pos.txt", "r", encoding='UTF-8') as s:  # 加载正面语料
    pos_data1 = s.readlines()

neg_data = sorted(set(neg_data1), key=neg_data1.index)  #列表去重 保持原来的顺序
pos_data = sorted(set(pos_data1), key=pos_data1.index)

neg_data = [del_stop_words(data.replace("\n", "")) for data in neg_data]   # 处理负面语料
pos_data = [del_stop_words(data.replace("\n", "")) for data in pos_data]
all_sentences = neg_data + pos_data  # 全部语料 用于训练word2vec

from gensim.models.word2vec import Word2Vec

import pickle

####用于训练词向量模型###

model = Word2Vec(all_sentences,     # 上文处理过的全部语料
                 vector_size=100,  # 词向量维度 默认100维
                 min_count=1,  # 词频阈值 词出现的频率 小于这个频率的词 将不予保存
                 window=5  # 窗口大小 表示当前词与预测词在一个句子中的最大距离是多少
                 )

index_dict, word_vectors= create_dictionaries(model)  # 索引字典、词向量字典
# 索引字典，{单词: 索引数字}
# 词向量, {单词: 词向量(100维长的数组)}

vocab_dim = 100

n_symbols = len(index_dict) + 1  # 索引数字的个数，因为有的词语索引为0，所以+1
embedding_weights = np.zeros((n_symbols, vocab_dim))  # 创建一个n_symbols * 100的0矩阵

for w, index in index_dict.items():  # 从索引为1的词语开始，用词向量填充矩阵
    embedding_weights[index, :] = word_vectors[w]  # 词向量矩阵，第一行是0向量（没有索引为0的词语，未被填充）


data = all_sentences  #获取之前分好词的数据
label_list = ([0] * len(neg_data) + [1] * len(pos_data))
# 划分训练集和测试集，此时都是list列表
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(data, label_list, test_size=0.2)

X_train = text_to_index_array(index_dict, X_train_l)
X_test = text_to_index_array(index_dict, X_test_l)

y_train = np.array(y_train_l)  # 转numpy数组
y_test = np.array(y_test_l)

##将数据切割成一样的指定长度
from torch.nn.utils.rnn import pad_sequence
#将数据补长变成和最长的一样长
X_train = pad_sequence([torch.from_numpy(np.array(x)) for x in X_train],batch_first=True).float()
X_test = pad_sequence([torch.from_numpy(np.array(x)) for x in X_test],batch_first=True).float()
#将数据切割成需要的样子
X_train = text_cut_to_same_long(X_train)
X_test = text_cut_to_same_long(X_test)

#将词向量字典序号转换为词向量矩阵
X_train = creat_wordvec_tensor(embedding_weights, X_train)
X_test = creat_wordvec_tensor(embedding_weights, X_test)

torch.save([torch.tensor(X_train), torch.tensor(y_train)], 'data/train/train.buffer')
torch.save([torch.tensor(X_test), torch.tensor(y_test)], 'data/test/test.buffer')
