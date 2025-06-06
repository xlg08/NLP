from utils import *


def my_getdata():
	# todo:1- 读取文件数据集, 得到 [[英文句子1, 法文句子1], [英文句子2, 法文句子2], ...]内存数据集
	# 1-1 with open 读取文件数据集
	with open(data_path, 'r', encoding='utf-8') as f:
		my_lines = f.read().strip().split('\n')
	# print('my_lines --->', my_lines)
	# 1-2 获取 [[英文句子1, 法文句子1], [英文句子2, 法文句子2], ...] 数据集格式
	# 定义两个空列表
	tmp_pair, my_pairs = [], []
	# 循环遍历my_lines
	for line in my_lines:
		# print('line--->', line)  # i m .	j ai ans .
		# 对my_lines中每行样本使用\t分割符进行分割后再循环遍历
		for item in line.split('\t'):
			# print('item--->', item)
			# 将每行样本中的英文句子和法文句子使用工具函数进行清洗, 保存到tmp_pair列表中
			tmp_pair.append(normalizeString(item))
		# 将tmp_pair列表保存到my_pairs列表中
		my_pairs.append(tmp_pair)
		# 重置tmp_pair列表
		tmp_pair = []
	# print('my_pairs的长度为--->', len(my_pairs))
	# print('my_pairs[:4]--->', my_pairs[:4])
	
	# todo:2-构建英文和法文词表 {词:下标} {下标:词}
	# 2-0: 初始化词表, 有SOS和EOS两个词
	english_word2index = {'SOS': 0, 'EOS': 1}
	# 定义第3个词起始下标
	english_word_n = 2
	french_word2index = {'SOS': 0, 'EOS': 1}
	french_word_n = 2
	
	# 2-1: 循环遍历my_pairs [['i m .', 'j ai ans .'], ...]
	for pair in my_pairs:
		# print('pair--->', pair)  # ['i m .', 'j ai ans .']
		# 2-2: 对英文句子或法文句子根据 ' '空格进行分割, 再进行循环遍历
		for word in pair[0].split(' '):
			# print('word--->', word)  # i  m  .
			# 2-3: 使用if语句, 判断当前词是否在词表中, 如果不在添加进去
			if word not in english_word2index.keys():
				english_word2index[word] = english_word_n
				# 更新词下标
				english_word_n += 1
		for word in pair[1].split(' '):
			# 2-3: 使用if语句, 判断当前词是否在词表中, 如果不在添加进去
			if word not in french_word2index.keys():
				french_word2index[word] = french_word_n
				# 更新词下标
				french_word_n += 1
	
	# 2-4 获取{下标:词}格式词表
	english_index2word = {v: k for k, v in english_word2index.items()}
	french_index2word = {v: k for k, v in french_word2index.items()}
	# print('english_word2index--->', len(english_word2index), english_word2index)
	# print('french_word2index--->', len(french_word2index), french_word2index)
	# print('english_index2word--->', len(english_index2word), english_index2word)
	# print('french_index2word--->', len(french_index2word), french_index2word)
	# print('english_word_n--->', english_word_n)
	# print('french_word_n--->', french_word_n)
	return english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n, my_pairs


# 自定义张量数据源类
class MyPairsDataset(Dataset):
	# todo:1- init构造方法, 初始化属性
	def __init__(self, my_pairs, english_word2index, french_word2index):
		self.my_pairs = my_pairs  # [[], [], ...]
		self.english_word2index = english_word2index
		self.french_index2word = french_word2index
		# 获取数据集长度
		self.sample_len = len(my_pairs)
	
	# todo:2- len方法, 返回数据集的长度
	def __len__(self):
		return self.sample_len
	
	# todo:3- getitem方法, 对数据进行处理, 转换成张量数据对象
	def __getitem__(self, index):
		"""
		转换成张量数据对象
		:param index: 数据集的下标 -> 第index个样本
		:return: tensor_x, tensor_y
		"""
		# 3-1: 修正index, 防止超过下标边界
		index = min(max(index, 0), self.sample_len - 1)
		# print('index--->', index)
		# 3-2: 获取当前index样本中的 x和y
		x = self.my_pairs[index][0]
		y = self.my_pairs[index][1]
		# print('x--->', x)
		# print('y--->', y)
		# 3-3: 将x和y的字符串数据转换成下标表示  词表
		# self.english_word2index[word]: 根据key获取字典中的value
		x = [self.english_word2index[word] for word in x.split(' ')]
		y = [self.french_index2word[word] for word in y.split(' ')]
		# print('x--->', x)
		# print('y--->', y)
		# 3-4: 每个样本最后加EOS下标 结束符号
		x.append(EOS_token)
		y.append(EOS_token)
		# print('x--->', x)
		# print('y--->', y)
		# 3-5: 将下标列表转换成张量对象
		# device: 将张量创建到对应的设备上 GPU/CPU
		tensor_x = torch.tensor(x, dtype=torch.long, device=device)
		tensor_y = torch.tensor(y, dtype=torch.long, device=device)
		# print('tensor_x--->', tensor_x)
		# print('tensor_y--->', tensor_y)
		return tensor_x, tensor_y


if __name__ == '__main__':
	(english_word2index, english_index2word, english_word_n,
	 french_word2index, french_index2word, french_word_n, my_pairs) = my_getdata()
	# 创建自定义数据源对象
	my_dataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
	print('my_dataset数据集条目数--->', len(my_dataset))
	print(my_dataset[0])
	# 创建数据加载器对象
	my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
	# 循环遍历数据加载器
	for i, (x, y) in enumerate(my_dataloader):
		print('x--->', x.shape, x)
		print('y--->', y.shape, y)
		break