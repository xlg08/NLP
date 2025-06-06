from decoderrnn import *

PATH1 = 'model/my_encoderrnn_2.pth'		# 编码器模型地址
PATH2 = 'model/my_attndecoderrnn_2.pth'		# 解码器模型地址

def seq2seq_evaluate(x,
                     my_encoderrnn: EncoderRNN,
                     my_attndecoderrnn: AttnDecoderRNN,
                     french_index2word):
	"""
		推理内部函数, 得到预测的法文
		:param x: 需要推理的英文句子
		:param my_encoderrnn: 编码器
		:param my_attndecoderrnn: 解码器
		:param french_index2word: 法文词汇表, 根据最大概率的下标从词表中获取法文词
		:return: 法文列表, 注意力权重概率矩阵
	"""
	with torch.no_grad():

		# 编码器与解码器设置为推理模式
		my_encoderrnn.eval()
		my_attndecoderrnn.eval()

		# todo: 1- 编码器编码
		encode_h0 = my_encoderrnn.inithidden()
		encode_output, encode_hn = my_encoderrnn(x, encode_h0)

		# todo: 2- 处理编码的输出 得到解码器的参数v
		encode_output_c = torch.zeros(size=(MAX_LENGTH, my_encoderrnn.hidden_size), device=device)
		for idx in range(x.shape[1]):
			encode_output_c[idx] = encode_output[0, idx]

		# todo: 3- 准备解码器的q和k参数
		decode_hidden = encode_hn				# 编码器的最后一个隐藏状态
		input_y = torch.tensor(data=[[SOS_token]], device=device)	 # 起始字符转化为二维张量，用于输入到解码器中，告诉解码器开始进行解码工作

		# todo: 4- 定义变量 预测词空列表
		decode_words = []

		# todo: 6- 创建(10,10)全0张量, 存储每个时间步的注意力权重
		# (10, 10) 		->	 	10:最多10个时间步 	10:权重概率矩阵特征数为10
		decoder_attentions = torch.zeros(size=(MAX_LENGTH, MAX_LENGTH), device=device)

		# todo: 7- 解码器解码
		for i in range(MAX_LENGTH):

			# 解码
			output_y, decode_hidden, attn_weights = my_attndecoderrnn(input_y, decode_hidden, encode_output_c)
			# print('attn_weights--->', attn_weights.shape, attn_weights)

			# 保存当前时间步的attn_weights
			decoder_attentions[i] = attn_weights
			# print('decoder_attentions--->', decoder_attentions.shape, decoder_attentions)

			# 获取当前时间步的预测结果 topv topi
			# topi = torch.argmax(output_y)
			topv, topi = output_y.topk(1)
			# 判断topi是否是EOS_token下标值
			# 如果是, 解码结束
			if topi.item() == EOS_token:
				decode_words.append('<EOS>')
				break
			else:
				decode_words.append(french_index2word[topi.item()])
			# 进行下一个时间步的预测
			input_y = topi

	# 返回法文列表, 注意力权重概率矩阵
	# [: i+1]->后续的值都为0,没有意义
	return decode_words, decoder_attentions[: i+1]


# 定义模型推理函数
def inference():

	# todo:1- 加载推理数据集
	(english_word2index, english_index2word, english_word_n,
	 french_word2index, french_index2word, french_word_n, my_pairs) = my_getdata()

	# todo:2- 创建张量数据集
	my_dataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)

	# todo:3- 加载模型
	# 编码器模型对象
	my_encoderrnn = EncoderRNN(input_size=english_word_n, hidden_size=256)
	'''
		my_encoderrnn---> EncoderRNN(
			(embedding): Embedding(2803, 256)
			(gru): GRU(256, 256, batch_first=True)
		)
		这个返回的是 PyTorch 中的 nn.Module 子类（定义的 EncoderRNN）的实例对象，当打印它时，
			会自动调用 __repr__() 方法，返回一个可读的模块结构描述，用于展示模型的层次结构。
	'''
	# print('my_encoderrnn--->', my_encoderrnn)

	# 加载模型参数
	# 参数：map_location: 将模型加载到什么设备中
	# 		lambda storage,loc: storage:保存时在哪个设备,加载就在哪个设备
	# 参数：strict: 是否严格按照创建时键值匹配加载 -> init方法中gru层属性名
	# 		参数值为 True 时: 匹配不成功, 报错
	# 		参数值为 False 时: 不报错, 但是不执行不匹配的层
	my_encoderrnn.load_state_dict(torch.load(PATH1, map_location=lambda storage, loc: storage), strict=False)
	# print('my_encoderrnn--->', my_encoderrnn)

	# 解码器模型对象
	my_attndecoderrnn = AttnDecoderRNN(output_size=french_word_n, hidden_size=256)
	my_attndecoderrnn.load_state_dict(torch.load(PATH2, map_location=lambda storage, loc: storage), strict=False)

	# todo:4- 准备3条测试样本
	my_samplepairs = [['i m impressed with your french .', 'je suis impressionne par votre francais .'],
	                  ['i m more than a friend .', 'je suis plus qu une amie .'],
	                  ['she is beautiful like her mother .', 'elle est belle comme sa mere .']]
	# print('my_samplepairs(样本数量) --->', len(my_samplepairs))

	# todo:5- 对测试样本进行处理, 训练时怎么做特征工程,推理时一样
	for idx, pair in enumerate(my_samplepairs):

		x = pair[0]			# 取出英语句子
		y = pair[1]			# 取出法语句子
		# print('x--->', x)
		# print('y--->', y)

		# 对x转换成下标张量对象
		tem_x = [english_word2index[word] for word in x.split(' ')]			# 将本次的英文句子根据英语词表转化为词索引列表
		tem_x.append(EOS_token)		# 将结束字符，放入到句子索引列表中
		print("x 索引列表 ：", tem_x)
		tensor_x = torch.tensor([tem_x], dtype=torch.long, device=device)
		print('tensor_x--->', tensor_x)
		print('tensor_x.shape ---> ', tensor_x.shape)			# seq_len

		# todo:6- 调用内部封装推理函数,进行推理
		decode_words, decoder_attentions = seq2seq_evaluate(tensor_x, my_encoderrnn, my_attndecoderrnn, french_index2word)
		# print('decode_words--->', decode_words)
		# print('decoder_attentions--->', decoder_attentions.shape, decoder_attentions)

		# todo:7- 将预测的法文列表转换成字符串文本
		output_sentence = ' '.join(decode_words)

		print('\n')
		print('需要推理的英文句子--->', x)
		print('真实的法文句子--->', y)
		print('推理的法文句子--->', output_sentence)


if __name__ == '__main__':
	inference()
