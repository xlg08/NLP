'''

'''
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
import jieba
import torch.nn as nn


# 1.对句子分词 word_list
# 2.对句子word2id求my_token_list，对句子文本数值化sentence2id
# 3.创建nn.Embedding层，查看每个token的词向量数据
# 4.从nn.Embedding层中根据idx拿词向量

def dm02_nnembeding_show():
    # 1 对句子分词 word_list
    sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    sentence2 = "我爱自然语言处理"
    sentences = [sentence1, sentence2]

    word_list = []

    # 分词
    for sentence in sentences:
        word_list.append(jieba.lcut(sentence))
    # print('word_list--->', word_list)

    # 2 对句子 word2id 求 my_token_list，对句子文本数值化 sentence2id
    mytokenizer = Tokenizer()
    mytokenizer.fit_on_texts(texts=word_list)
    # print(mytokenizer.index_word, mytokenizer.word_index)

    # 打印my_token_list
    my_token_list = mytokenizer.index_word.values()
    # print('my_token_list-->', my_token_list)

    # 打印文本数值化以后的句子
    sentence2id = mytokenizer.texts_to_sequences(texts=word_list)
    # print('sentence2id--->', sentence2id, len(sentence2id))

    # 3 创建nn.Embedding层
    embd = nn.Embedding(num_embeddings=len(my_token_list), embedding_dim=8)
    # print("embd--->", embd)
    # print('nn.Embedding层词向量矩阵-->', embd.weight.data, embd.weight.data.shape, type(embd.weight.data))

    #  4 从nn.Embedding层中根据idx拿词向量
    for idx in range(len(mytokenizer.index_word)):
        tmpvec = embd(torch.tensor(idx))
        print(tmpvec)
        # print('%4s' % (mytokenizer.index_word[idx + 1]), tmpvec.detach().numpy())


# 词嵌入
def demo_emb():
    # 文本
    sentence1 = '传智教育是一家上市公司，旗下有黑马程序员品牌。我是在黑马这里学习人工智能'
    sentence2 = "我爱自然语言处理"
    sentences = [sentence1, sentence2]
    word_list = []

    # 分词
    for sentence in sentences:
        word_list.append(jieba.lcut(sentence))
    print(word_list)

    # Tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(word_list)
    print(tokenizer.word_index)
    print(tokenizer.index_word)

    # 获取所有token
    word_list = tokenizer.index_word.values()

    # 词嵌入层:词表大小，词向量维度
    embed = nn.Embedding(num_embeddings=len(word_list), embedding_dim=5)

    # 获取特定词语的向量
    print(embed(torch.tensor(19 - 1)))
    
    # 遍历获取每一个词的词向量
    for i in range(len(word_list)):  # 词语
        # 对应的词语的id
        id = tokenizer.word_index[list(word_list)[i]]
        print(list(word_list)[i])
        # 获取嵌入的结果
        print(embed(torch.tensor(id - 1)))


if __name__ == '__main__':
    dm02_nnembeding_show()
