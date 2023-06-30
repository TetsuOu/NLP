from gensim.corpora.dictionary import Dictionary
import numpy as np

maxlen = 28 # 文本保留的最大长度
vocab_dim = 100

# 加载模型，提取出词索引和词向量
def create_dictionaries(model):
    gensim_dict = Dictionary()  # 创建词语词典
    gensim_dict.doc2bow(model.wv.key_to_index.keys(), allow_update=True)

    w2indx = {v: k + 1 for k, v in gensim_dict.items()}  # 词语的索引，从1开始编号
    w2vec = {word: model.wv.get_vector(word) for word in w2indx.keys()}  # 词语的词向量
    return w2indx, w2vec

# 将文本数据映射成数字（是某个词的编号，不是词向量）
def text_to_index_array(p_new_dic, p_sen):
    ##文本或列表转换为索引数字

    if type(p_sen) == list:
        new_sentences = []
        for sen in p_sen:
            new_sen = []
            for id, word in enumerate(sen):
                try:
                    new_sen.append(p_new_dic[word])  # 单词转索引数字
                except:
                    new_sen.append(0)  # 索引字典里没有的词转为数字0
            new_sentences.append(new_sen)
        return np.array(new_sentences)  # 转numpy数组
    else:
        new_sentences = []
        sentences = []
        p_sen = p_sen.split(" ")
        for word in p_sen:
            try:
                sentences.append(p_new_dic[word])  # 单词转索引数字
            except:
                sentences.append(0)  # 索引字典里没有的词转为数字0
        new_sentences.append(sentences)
        return new_sentences


# 将数据切割成一样的指定长度
def text_cut_to_same_long(sents):
    data_num = len(sents)
    new_sents = np.zeros((data_num, maxlen))  # 构建一个矩阵来装修剪好的数据
    se = []
    for i in range(len(sents)):
        new_sents[i, :] = sents[i, :maxlen]
    new_sents = np.array(new_sents)
    return new_sents


# 将每个句子的序号矩阵替换成词向量矩阵
def creat_wordvec_tensor(embedding_weights, X_T):
    X_tt = np.zeros((len(X_T), maxlen, vocab_dim))
    num1 = 0
    num2 = 0
    for j in X_T:
        for i in j:
            X_tt[num1, num2, :] = embedding_weights[int(i), :]
            num2 = num2 + 1
        num1 = num1 + 1
        num2 = 0
    return X_tt
