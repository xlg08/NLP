import random

import matplotlib.pyplot as plt
import torch

from decoderrnn import *

# 模型训练参数
mylr = 1e-4
epochs = 2
# 设置teacher_forcing比率为0.5
teacher_forcing_ratio = 0.5
# 1000次迭代打印一次信息
print_interval_num = 1000
# 100次迭代绘制损失曲线
plot_interval_num = 100


def train_iters(x, y,
                my_encoderrnn: EncoderRNN,
                my_attndecoderrnn: AttnDecoderRNN,
                myadam_encode: optim.Adam,
                myadam_decode: optim.Adam,
                mynllloss: nn.NLLLoss):
	"""
	模型训练的内部函数 -> 内循环代码封装
	:param x: 英文句子
	:param y: 真实法文句子
	:param my_encoderrnn: 编码器
	:param my_attndecoderrnn:  解码器
	:param myadam_encode: 编码器优化器
	:param myadam_decode: 解码器优化器
	:param mynllloss: 解码器损失函数对象
	:return: 当前句子的平均损失
	"""
	# todo:1- 切换模型训练模式
	my_encoderrnn.train()
	my_attndecoderrnn.train()
	# todo:2- 初始化编码器隐藏状态值
	encode_h0 = my_encoderrnn.inithidden()
	# todo:3- 调用编码器获取v和k output就是v k就是解码器的初始隐藏状态值
	encode_output, encode_hn = my_encoderrnn(x, encode_h0)
	# print('encode_output--->', encode_output.shape, encode_output)
	# print('encode_hn--->', encode_hn.shape, encode_hn)
	# todo:4- 处理v, 统一长度, 都是10  v
	encode_output_c = torch.zeros(size=(MAX_LENGTH, my_encoderrnn.hidden_size), device=device)
	# print('encode_output_c--->', encode_output_c.shape, encode_output_c)
	for idx in range(x.shape[1]):
		encode_output_c[idx] = encode_output[0, idx]
	# print('encode_output_c--->', encode_output_c.shape, encode_output_c)
	# todo:5- 准备解码器第一个时间步的参数 q,k,v
	# 准备k
	decode_hidden = encode_hn
	# 准备q
	input_y = torch.tensor(data=[[SOS_token]], device=device)
	# print('input_y--->', input_y.shape, input_y)
	# todo:6- 初始化变量, 存储信息
	myloss = 0.0  # 当前句子的总损失
	iters_num = 0  # 当前句子的token数
	# todo:7- 判断教师强制机制是否成立, 返回True或False
	user_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
	# print('user_teacher_forcing--->', user_teacher_forcing)
	# todo:8- 解码器自回归解码
	# 预测什么时候结束? ①到达循环次数,法文句子长度 ②预测出EOS_token
	for idx in range(y.shape[1]):
		# 调用解码器模型对象, 返回预测token, 隐藏状态值, 注意力机制概率矩阵
		output_y, hidden, attn_weights = my_attndecoderrnn(input_y, decode_hidden, encode_output_c)
		# print('output_y--->', output_y.shape, output_y)
		# 获取当前时间步真实的token
		target_y = y[0][idx].view(1)
		# print('target_y--->', target_y.shape, target_y)
		# 计算损失值
		myloss += mynllloss(output_y, target_y)
		# print('myloss--->', myloss)
		# 更新iters_num数
		iters_num += 1
		# print('iters_num--->', iters_num)
		# 使用教师强制机制, 判断下一时间步使用真实token还是预测token
		if user_teacher_forcing:
			# input_y = y[0][idx].view(1, -1)
			input_y = target_y.view(1, -1)
		# print('input_y--->', input_y.shape, input_y)
		else:
			# 返回最大值和对应的下标
			topv, topi = output_y.topk(1)
			# print('topv--->', topv)
			# print('topi--->', topi)
			# 预测出结束符号, 解码结束
			if topi.item() == EOS_token:
				break
			input_y = topi
	
	# todo:9-梯度清零, 反向传播, 梯度更新
	myadam_encode.zero_grad()
	myadam_decode.zero_grad()
	
	myloss.backward()
	
	myadam_encode.step()
	myadam_decode.step()
	
	# todo:10- 句子的平均损失  总损失/token数
	return myloss.item() / iters_num


def train_seq2seq():
	# 获取数据
	(english_word2index, english_index2word, english_word_n,
	 french_word2index, french_index2word, french_word_n, my_pairs) = my_getdata()
	# 实例化 mypairsdataset对象  实例化 mydataloader
	mypairsdataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)
	mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)
	
	# 实例化编码器 my_encoderrnn 实例化解码器 my_attndecoderrnn
	my_encoderrnn = EncoderRNN(english_word_n, 256).to(device)
	my_attndecoderrnn = AttnDecoderRNN(output_size=french_word_n, hidden_size=256, dropout_p=0.1, max_length=10).to(
		device)
	
	# 实例化编码器优化器 myadam_encode 实例化解码器优化器 myadam_decode
	myadam_encode = optim.Adam(my_encoderrnn.parameters(), lr=mylr)
	myadam_decode = optim.Adam(my_attndecoderrnn.parameters(), lr=mylr)
	
	# 实例化损失函数 mycrossentropyloss = nn.NLLLoss()
	mynllloss = nn.NLLLoss()
	
	# 定义模型训练的参数
	plot_loss_list = []
	
	# 循环轮次 epoch
	for epoch_idx in range(1, epochs + 1):
		# 初始化打印日志的总损失 和 绘图总损失
		print_loss_total = 0.0
		plot_loss_total = 0.0
		# 开始时间
		starttime = time.time()
		# 循环迭代次数, batch数
		# start: 默认为0, 第一条数据的标号为0; 1->第一条数据的标号为1
		for item, (x, y) in enumerate(mydataloader, start=1):
			# 模型训练, 调用内部迭代函数
			loss = train_iters(x, y,
			                   my_encoderrnn,
			                   my_attndecoderrnn,
			                   myadam_encode,
			                   myadam_decode,
			                   mynllloss)
			# print('loss--->', loss)
			# 统计损失值
			print_loss_total += loss
			plot_loss_total += loss
			# 1000次迭代打印一次日志
			if item % 1000 == 0:
				print_loss_avg = print_loss_total / print_interval_num
				# 重置print_loss_total 0
				print_loss_total = 0.0
				# 打印日志，日志内容分别是：训练耗时，当前迭代步，当前进度百分比，当前平均损失
				print('轮次%d  损失%.6f 时间:%d' % (epoch_idx, print_loss_avg, time.time() - starttime))
			# 100次收集一次损失, 用于绘图
			if item % 100 == 0:
				plot_loss_avg = plot_loss_total / plot_interval_num
				plot_loss_list.append(plot_loss_avg)
				plot_loss_total = 0.0
		torch.save(my_encoderrnn.state_dict(), './model/my_encoderrnn_model_%d.bin' % (epoch_idx))
		torch.save(my_attndecoderrnn.state_dict(), './model/my_attndecoderrnn_model_%d.bin' % (epoch_idx))
	
	# 绘制损失值的曲线图
	plt.figure()
	# plt.plot(plot_loss_list.detach().numpy())
	plt.plot(plot_loss_list.numpy())
	plt.savefig('./image/plot_loss_list.png')
	plt.show()
	return plot_loss_list


if __name__ == '__main__':
	plot_loss_list = train_seq2seq()
