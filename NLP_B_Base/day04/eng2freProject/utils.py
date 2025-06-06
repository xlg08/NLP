# 用于正则表达式
import re
# 用于构建网络结构和函数的torch工具包
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# torch中预定义的优化方法工具包
import torch.optim as optim
import time
# 用于随机生成数据
import random
import numpy as np
import matplotlib.pyplot as plt

# 定义变量
# 选择设备 cpu/gpu
# 'cuda'->使用所有显卡  'cuda:0'->使用第一张显卡
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 起始符号下标
# sos -> start of sentences
SOS_token = 0
# 结束符号下标
EOS_token = 1
# 文件路径
data_path = 'data/eng-fra-v2.txt'
# 最大句子长度, 预处理分析的结果
MAX_LENGTH = 10


# 定义处理文本的工具函数  处理句子中的特殊符号/大小写/换行符
def normalizeString(s: str):
	# 转换成小写, 并删掉两端的空白符号
	str = s.lower().strip()
	# 正则表达式匹配标签符号'.?!' 转换成 ' .?!'
	str = re.sub(r'([.!?])', r' \1', str)
	# print('str--->', str)
	# 正则表达式匹配除a-z.!?之外的其他的符号 转换成 ' '
	str = re.sub(r'[^a-z.!?]+', r' ', str)
	# print('str--->', str)
	return str


if __name__ == '__main__':
	str1 = 'I m sad.@'
	normalizeString(str1)
