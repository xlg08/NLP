'''

'''
class Tokenizer():
    def __init__(self):
        self.word2idx = {
            "<pad>": 0,     # 填充符
            "<SOS>": 1,         # 句子开始符
            "<EOS>": 2          # 句子结束符
        }
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    # 根据每一个句子样本构建词汇表
    # 遍历出每一条句子中的每一个词，放到列表(词汇表)中，并判断该词在词汇中存不存在，如果不存在再放到词汇表中，因为词汇表中的词汇是不能重复的
    def build_vocab(self, sentences):       # 参数 sentences 为 句子列表
        for sentence in sentences:      # 遍历出每一条句子
            for word in sentence.split():       # 每一条句子根据空格进行分词
                if word not in self.word2idx:       # 默认判断的是不在字典的键中
                    self.word2idx[word] = len(self.word2idx)        # 根据词汇表




