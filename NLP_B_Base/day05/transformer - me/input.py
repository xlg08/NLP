# 输入部分是由 词嵌入层和位置编码层组成   x = word_embedding + position_encoding
import torch
import torch.nn as nn
import math


# 词嵌入层
class Embeddings(nn.Module):
    # todo:1- 定义构造方法 init
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # 初始化属性
        self.vocab = vocab_size  # 词表大小
        self.d_model = d_model  # 词向量维度
        # 初始化词嵌入层对象
        # padding_idx: 将值为0的值, 不进行词向量, 用0填充
        self.embedding = nn.Embedding(num_embeddings=self.vocab,
                                      embedding_dim=self.d_model,
                                      padding_idx=0)

    # todo:2- 定义forward方法 前向传播
    def forward(self, x):
        # 词嵌入结果乘以根号维度数
        # 最终的词向量值和后续位置编码信息差不太多, 实现信息平衡
        # 后续注意力机制使用的缩放点积, 乘和除相抵消
        return self.embedding(x) * math.sqrt(self.d_model)


# 位置编码器
class PositionalEncoding(nn.Module):
	# todo:1- init方法
	def __init__(self, d_model, max_len=5000, dropout_p=0.1):

		super().__init__()
		self.d_model = d_model
		self.max_len = max_len
		self.dropout = nn.Dropout(p=dropout_p)

		# 初始化公式所需要的值
		pos = torch.arange(0, self.max_len).unsqueeze(1)		# (1, 1)
		pe = torch.zeros(size=(self.max_len, self.d_model))		# (token, 词向量)
		_2i = torch.arange(0, self.d_model, 2).float()

		pe[:, ::2] = torch.sin(pos / 10000 ** (_2i / self.d_model))
		pe[:, 1::2] = torch.cos(pos / 10000 ** (_2i / self.d_model))

		pe = pe.unsqueeze(0)

		self.register_buffer('pe', pe)

	def forward(self, x):

		return x + self.pe[:, :x.shape[1], :]