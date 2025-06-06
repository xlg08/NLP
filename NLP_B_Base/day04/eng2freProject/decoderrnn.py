import torch

from encoderrnn import *


class DecoderRNN(nn.Module):
	# todo:1- 定义构造方法 init
	def __init__(self, output_size, hidden_size):
		super().__init__()
		# 初始化法文词表大小维度属性=线性输出层的维度
		self.output_size = output_size
		# 初始化gru隐藏层和词嵌入层的维度属性  共用
		self.hidden_size = hidden_size
		# 初始化词嵌入层
		# num_embeddings: 法文词表大小
		# embedding_dim: 词向量初始维度
		self.embeding = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.hidden_size)
		# 初始化gru层
		self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
		
		# 初始化全连接层 线性层+激活层
		# out_features: 法文词表大小  预测出n个词的生成概率
		self.out = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
		# dim:一定是-1, 按行处理
		self.softmax = nn.LogSoftmax(dim=-1)
	
	# todo:2- 定义前向传播方法 forward
	def forward(self, input, hidden):
		print('input--->', input.shape)
		# 词嵌入操作
		embedded = self.embeding(input)
		print('embedded--->', embedded.shape)
		# 通过relu激活函数引入非线性因素, 防止过拟合(x<0置为0, 神经元死亡)
		embedded = torch.relu(embedded)
		print('embedded--->', embedded.shape)
		# gru层操作
		# ouput: 输入input的语义信息, 形状为(句子数, 句子长度, 词维度) 三维
		output, hidden = self.gru(embedded, hidden)
		print('output--->', output.shape, output)
		# 全连接层操作
		# output[0]: 全连接层一般是二维数据, 所以要取出当前token的二维表示
		# 返回的output是 logsoftmax结果, 后续的值可能会有负值, 不是softmax的概率值
		output = self.softmax(self.out(output[0]))
		print('output--->', output.shape, output)
		return output, hidden


# 带加性注意力机制的解码器
class AttnDecoderRNN(nn.Module):
	# todo:1- 定义构造方法 init
	def __init__(self, output_size, hidden_size, dropout_p=0.2, max_length=MAX_LENGTH):
		super().__init__()
		# 初始化词嵌入层的输入维度和全连接层的输出维度一致
		self.output_size = output_size
		# 初始化编码器解码器隐藏层维度属性  解码器的第一个隐藏状态值=编码器的最后一个隐藏状态值
		# 初始化词嵌入层维度属性  共享
		self.hidden_size = hidden_size
		# 初始化最大句子长度属性 -> 所有句子 c的长度固定
		self.max_length = max_length
		# 初始化dropout概率属性
		self.dropout_p = dropout_p
		# 初始化 embeding层
		self.embedding = nn.Embedding(num_embeddings=self.output_size, embedding_dim=self.hidden_size)
		# 初始化 gru层
		self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)
		# 初始化 全连接层
		self.out = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)
		self.softmax = nn.LogSoftmax(dim=-1)
		
		# 初始化注意力机制中两个线性层
		"""
		q:解码器当前预测时间步的隐藏状态值
		k:解码器当前预测时间步的上一时间步隐藏状态值
		v:编码器的output输出
		q,k,v三个特征维度相同 都是hidden_size
		"""
		# in_features: q和k的特征维度拼接
		# out_features: 后续权重概率矩阵->(1, 1, max_len) 和 V矩阵相乘 V->(1, max_len, hidden_size)
		self.attn = nn.Linear(in_features=self.hidden_size + self.hidden_size, out_features=self.max_length)
		# in_features: q和c的特征维度拼接
		# out_features: 输出的维度和gru层的输入维度保持一致
		self.attn_combine = nn.Linear(in_features=self.hidden_size + self.hidden_size, out_features=self.hidden_size)
		# 初始化dropout层
		self.dropout = nn.Dropout(p=self.dropout_p)
	
	# todo:2- 定义前向传播方法 forward
	def forward(self, input, hidden, encoder_outputs):
		"""
		前向传播计算
		:param input: q, 解码器当前预测时间步的输入x, 也是上一个时间步预测的输出y
		:param hidden: k, 上一个时间步的隐藏状态值, 第一个时间步的上一个隐藏状态值=编码器最后一个时间步的隐藏状态值
		:param encoder_outputs: v, 编码器的输出 output, 后续是统一长度都为10, 10个token, 不足10个token用0填充
		:return: 预测词表概率向量, 当前时间步的隐藏状态值, 权重概率矩阵
		"""
		# 2-1 词嵌入操作
		embedded = self.embedding(input)
		# 使用dropout防止过拟合
		embedded = self.dropout(embedded)
		# print('embedded--->', embedded.shape, embedded)
		
		# 2-2 计算权重分数矩阵, 之后再计算权重概率矩阵
		# q和k在特征维度轴拼接 + 线性计算 + softmax计算
		# embedded[0]: 获取二维向量表示, 线性层一般接收二维数据
		attn_weights = torch.softmax(self.attn(torch.cat(tensors=[embedded[0], hidden[0]], dim=1)), dim=-1)
		# print('attn_weights--->', attn_weights.shape, attn_weights)
		# print(torch.sum(input=attn_weights))
		
		# 2-3 计算动态c, 加权求和  权重概率矩阵和v进行三维矩阵乘法
		# bmm() 三维矩阵乘法, 目前attn_weights和encoder_outputs二维矩阵
		attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
		# print('attn_applied--->', attn_applied.shape, attn_applied)
		
		# 2-4 q和动态c融合线性计算, 得到gru的输入x
		# unsqueeze():得到三维数据, gru的输入x的形状要求
		output = self.attn_combine(torch.cat(tensors=[embedded[0], attn_applied[0]], dim=1)).unsqueeze(0)
		# print('output--->', output.shape, output)
		# relu激活函数, 非线性因素
		output = torch.relu(output)
		
		# 2-5 gru层操作
		output, hidden = self.gru(output, hidden)
		# print('output--->', output.shape, output)
		# print('hidden--->', hidden.shape, hidden)
		
		# 2-6 全连接层操作
		output = self.softmax(self.out(output[0]))
		# print('output--->', output.shape, output)
		return output, hidden, attn_weights


if __name__ == '__main__':
	# 获取数据
	(english_word2index, english_index2word, english_word_n,
	 french_word2index, french_index2word, french_word_n, my_pairs) = my_getdata()
	# 创建张量数据集
	my_dataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
	# 创建数据加载器
	# batch_size: 当前设置为1, 因为句子长度不一致
	my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
	# 创建编码器对象
	my_encoderrnn = EncoderRNN(input_size=english_word_n, hidden_size=256).to(device=device)
	# 创建解码器对象
	output_size = french_word_n
	hidden_size = 256
	my_decoderrnn = DecoderRNN(output_size, hidden_size).to(device)
	
	# 创建带attn的解码器对象
	my_attndecoderrnn = AttnDecoderRNN(output_size, hidden_size).to(device)
	for i, (x, y) in enumerate(my_dataloader):
		# print('x--->', x.shape)
		# 编码器进行编码 一次性喂数据
		# 初始化隐藏状态值
		hidden = my_encoderrnn.inithidden()
		encoder_output, hn = my_encoderrnn(x, hidden)
		print('encoder_output--->', encoder_output.shape, encoder_output)
		# print('hn--->', hn.shape, hn)
		
		# 获取填充成最大程度的编码器c或者output
		# 初始化全0的张量 形状(10, 256) [[0,0,0,0,0,0,...],[],[]]
		encoder_output_c = torch.zeros(size=(MAX_LENGTH, my_encoderrnn.hidden_size), device=device)
		# 将encoder_output真实值赋值到encoder_output_c对应位置
		for idx in range(x.shape[1]):
			encoder_output_c[idx] = encoder_output[0][idx]
		print('encoder_output_c--->', encoder_output_c.shape, encoder_output_c)
		# 解码器进行解码, 自回归, 一个一个token进行解码
		for j in range(y.shape[1]):
			# 获取当前预测token时间步的输入x(等同于上一时间步的预测y)
			# 当前以真实y中的每个token作为输入, 模拟解码器的界面过程, 实际上第一个输入token一定是起始符号
			tmp_y = y[0][j].view(1, -1)
			# 进行解码
			# 初始的隐藏状态值=编码器最后一个时间步的隐藏状态值
			# my_decoderrnn(tmp_y, hn)
			# hn:编码器端最后一个时间步的隐藏状态值, 也是解码器端第一个时间步的初始的隐藏状态值
			# print('hn--->', hn.shape, hn)
			output, hidden, attn_weights = my_attndecoderrnn(tmp_y, hn, encoder_output_c)
			print('=' * 80)
			print('output--->', output.shape, output)
			print('hidden--->', hidden.shape, hidden)
			print('attn_weights--->', attn_weights.shape, attn_weights)
			break
		break
