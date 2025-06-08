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
		# 初始化属性
		self.d_model = d_model  # 词向量维度, 模型维度
		self.max_len = max_len  # 最大句子长度
		self.dropout = nn.Dropout(p=dropout_p)

		# 获取句子所有token索引下标,作为pos
		# .unsqueeze(1)->在1轴升维 [[0],[1],...]
		pos = torch.arange(0, self.max_len).unsqueeze(1)
		# print('pos--->', pos)
		# print('pos.shape--->', pos.shape)

		# 创建一个pe全0矩阵, 存储位置信息  形状(最大句子长度, 词维度)
		pe = torch.zeros(size=(self.max_len, self.d_model))
		# print('pe--->', pe)
		# print('pe.shape--->', pe.shape)

		# 获取2i结果, 对向量维度d_model取偶数下标值
		_2i = torch.arange(0, self.d_model, 2).float()
		# print('_2i--->', _2i)		# [0, 2, 4, 6, ..., 510]
		# print('_2i--->', _2i.shape)

		# 计算位置信息 词向量维度奇数位的词 sin
		pe[:, ::2] = torch.sin(pos / 10000 ** (_2i / self.d_model))
		# print("sin维度：", torch.sin(pos / 10000 ** (_2i / self.d_model)).shape)
		# print("词向量偶数位置：", pe[:, ::2].shape)
		# 词向量维度偶数位的词 cos
		pe[:, 1::2] = torch.cos(pos / 10000 ** (_2i / self.d_model))

		# print("pe", pe)
		# print("pe.shape", pe.shape)		# seq_len, 词向量维度

		# 将pe位置矩阵升维, 三维数据集
		pe = pe.unsqueeze(0)			# 添加 batch_size 维度
		# print('pe--->', pe)
		# print('pe.shape--->', pe.shape)		# batch_size, seq_len, 词向量维度

		# 存储到内存中, 后续便于加载
		# pe属性中存储的是pe矩阵结果
		self.register_buffer('pe', pe)

	# todo:2- forward方法 将位置信息添加到词嵌入结果中
	def forward(self, x):
		"""
		:param x: 词嵌入层的输出结果
		:return: 编码器的输入x
		"""
		# print('x--->', x)
		# print('x.shape--->', x.shape)		# batch_size, seq_len, 词向量维度
		# x.shape[1], 句子中有多少个真实的token, 就需要在pe矩阵中取前多少个位置信息就行
		# print('x.shape[1]--->', x.shape[1])
		# print('self.pe[:, :x.shape[1], :]--->', self.pe[:, :x.shape[1], :].shape, self.pe[:, :x.shape[1], :])
		return x + self.pe[:, :x.shape[1], :]			# 只将词向量维度中的值进行相加，本质上就是将语义信息与位置信息相加


if __name__ == '__main__':

	vocab_size = 1000
	d_model = 512

	# 创建测试数据
	x = torch.LongTensor([[100, 2, 421, 508], [491, 999, 1, 0]])

	# 创建词嵌入对象
	my_embedding = Embeddings(vocab_size, d_model)

	# 调用对象实现词嵌入
	embedded_result = my_embedding(x)
	# print('embedded_result--->', embedded_result)
	# print('embedded_result.shape--->', embedded_result.shape)

	# 创建pe位置矩阵 生成位置特征数据[1,60,512]		# 句子数, 最大句子长度, 词向量的维度
	my_pe = PositionalEncoding(d_model=d_model, dropout_p=0.1, max_len=60)

	# 调用位置编码对象
	pe_result = my_pe(embedded_result)
	print('pe_result--->', pe_result.shape, pe_result)