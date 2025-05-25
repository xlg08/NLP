'''

'''

# 导入torch工具
import torch
# 导入nn准备构建模型
import torch.nn as nn
import torch.optim as optim
# 导入torch的数据源 数据迭代器工具包
from torch.utils.data import Dataset, DataLoader
# 用于获得常见字母及字符规范化
import string
# 导入时间工具包
import time
# 引入制图工具包
import matplotlib.pyplot as plt


# 获取所有常用字符包括字母和常用标点
all_letters = string.ascii_letters + " .,;'"
# 获取常用字符数量
n_letters = len(all_letters)
# print('all_letters--->', all_letters)
# print("n_letter:", n_letters)


# 国家名 种类数
categories = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
              'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
# 国家名 个数
n_categories = len(categories)
# print('categories--->', categories)
# print('n_categories--->', n_categories)

# 思路分析
# 1 打开数据文件 open(filename, mode='r', encoding='utf-8')
# 2 按行读文件、提取样本x 样本y line.strip().split('\t')
# 3 返回样本x的列表、样本y的列表 my_list_x, my_list_y
def read_data(filename):
    my_list_x, my_list_y = [], []
    # 打开文件
    with open(filename, mode='r', encoding='utf-8') as f:
        # 按照行读数据
        for line in f.readlines():
            if len(line) <= 5:
                continue
            # 按照行提取样本x 样本y
            x, y = line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)
    # 打印样本的数量
    print('my_list_x->', len(my_list_x))
    print('my_list_y->', len(my_list_y))
    # 返回样本x的列表、样本y的列表
    return my_list_x, my_list_y

# 原始数据 -> 数据源NameClassDataset --> 数据迭代器DataLoader
# 构造数据源 NameClassDataset，把语料转换成x y
# 1 init函数 设置样本x和y self.my_list_x self.my_list_y 条目数self.sample_len
# 2 __len__(self)函数  获取样本条数
# 3 __getitem__(self, index)函数 获取第几条样本数据
#       按索引 获取数据样本 x y
#       样本x one-hot张量化 tensor_x[li][all_letters.find(letter)] = 1
#       样本y 张量化 torch.tensor(categorys.index(y), dtype=torch.long)
#       返回tensor_x, tensor_y
class NameClassDataset(Dataset):
    def __init__(self, my_list_x, my_list_y):
        # 样本x
        self.my_list_x = my_list_x
        # 样本y
        self.my_list_y = my_list_y
        # 样本条目数
        self.sample_len = len(my_list_x)

    # 获取样本条数
    def __len__(self):
        return self.sample_len

    # 获取第几条 样本数据
    def __getitem__(self, index):
        # 对index异常值进行修正 [0, self.sample_len-1]
        index = min(max(index, 0), self.sample_len - 1)
        # 按索引获取 数据样本 x y
        x = self.my_list_x[index]
        y = self.my_list_y[index]
        # 样本x one-hot张量化
        tensor_x = torch.zeros(len(x), n_letters)
        # 遍历人名的每个字母做成one-hot编码
        for li, letter in enumerate(x):
            # letter2index 使用all_letters.find(letter)查找字母在all_letters表中的位置 给one-hot赋值
            tensor_x[li][all_letters.find(letter)] = 1
        # 样本y 张量化
        tensor_y = torch.tensor(categories.index(y), dtype=torch.long)
        # 返回结果
        return tensor_x, tensor_y

def dm_test_NameClassDataset():
    # 1 获取数据
    myfilename = './data/name_classfication.txt'
    my_list_x, my_list_y = read_data(myfilename)
    print('my_list_x length', len(my_list_x))
    print('my_list_y length', len(my_list_y))
    # 2 实例化dataset对象
    nameclassdataset = NameClassDataset(my_list_x, my_list_y)
    # 3 实例化dataloader
    mydataloader = DataLoader(dataset=nameclassdataset, batch_size=1, shuffle=True)
    for i, (x, y) in enumerate(mydataloader):
        print('x.shape', x.shape, x)
        print('y.shape', y.shape, y)
        break

# RNN类 实现思路分析：
# 1 init函数 准备三个层 self.rnn self.linear self.softmax=nn.LogSoftmax(dim=-1)
    # def __init__(self, input_size, hidden_size, output_size, num_layers=1)
# 2 forward(input, hidden)函数
    # 让数据经过三个层 返回softmax结果和hn
    # 形状变化 [seqlen,1,57],[1,1,128]) -> [seqlen,1,128],[1,1,128]
# 3 初始化隐藏层输入数据 inithidden()
    # 形状[self.num_layers, 1, self.hidden_size]
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        # 1 init函数 准备三个层 self.rnn self.linear self.softmax=nn.LogSoftmax(dim=-1)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # 定义rnn层
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers)
        # 定义linear层
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # 定义softmax层
        # LogSoftmax+NLLLoss=CrossEntropyLoss
        # 后续多分类交叉熵损失函数使用NLLLoss
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        '''input [6,57]->2维矩阵,表示1个人名样本 hidden[1,1,128]->3维矩阵'''
        # 数据形状 [6,57]->[6,1,57]->表示1个人名样本的数据集 1个句子数=1个人名 6个词=1个人名由6个字符组成 57=每个字符由57维向量表示
        # x形状->(句子长度, 词向量维度) 在1轴增加形状为1的句子数 (句子长度, 句子数, 词向量维度)
        input = input.unsqueeze(dim=1)
        # 1 数据经过模型 提取事物特征
        # 数据形状 [seqlen,1,57],[1,1,128]->[seqlen,1,128],[1,1,128]
        rr, hn = self.rnn(input, hidden)
        # 数据形状 [seqlen,1,128]->[1,128]  eg:[6,1,128]->[1,128]
        # 获取rr中最后一个时间步的输出
        tmprr = rr[-1]
        # 2 数据经过全连接层 [1,128]->[1,18] 18=国家各种类数=18个类别
        tmprr = self.linear(tmprr)
        # 3 数据经过softmax层返回
        output = self.softmax(tmprr)
        return output, hn

    def inithidden(self):
        # 初始化隐藏层输入数据 inithidden()
        return torch.zeros(size=(self.num_layers, 1, self.hidden_size))

def dm01_test_myrnn():
    # 1 实例化rnn对象
    my_rnn = RNN(input_size=57, hidden_size=128, output_size=18)
    # 2 准备数据 1个人名样本 2维矩阵
    # 后续经过forward中的input = input.unsqueeze(1), 转换成1个人名样本的数据集 (句子长度, 句子数, 词向量维度)
    input = torch.randn(size=(6, 57))
    print(input.shape)
    hidden = my_rnn.inithidden()
    # 3 给模型1次性的送数据
    # [seqlen,57], [1,1,128]) -> [1,18], [1,1,128]
    output, hidden = my_rnn(input, hidden)
    print('一次性的送数据：output->', output.shape, output)
    print('hidden->', hidden.shape)
    # 4 给模型1个字符1个字符的喂数据
    hidden = my_rnn.inithidden()
    for i in range(input.shape[0]):
        # input[i]: 获取第i个字符, 降维, 例如:[1,2,3]->1
        # unsqueeze(0):恢复维度 1->[1]
        tmpinput = input[i].unsqueeze(0)
        output, hidden = my_rnn(tmpinput, hidden)
    # 最后一次ouput
    print('一个字符一个字符的送数据output->', output.shape, output)


# LSTM类 实现思路分析：
# 1 init函数 准备三个层 self.rnn self.linear self.softmax=nn.LogSoftmax(dim=-1)
# def __init__(self, input_size, hidden_size, output_size, num_layers=1)
# 2 forward(input, hidden)函数
# 让数据经过三个层 返回softmax结果和hn
# 形状变化 [seqlen,1,57],[1,1,128]) -> [seqlen,1,128],[1,1,128]
# 3 初始化隐藏层输入数据 inithidden()
# 形状[self.num_layers, 1, self.hidden_size]
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        # 1 init函数 准备三个层 self.rnn self.linear self.softmax=nn.LogSoftmax(dim=-1)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # 定义rnn层
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        # 定义linear层
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, c):
        # 让数据经过三个层 返回softmax结果和 hn c
        # 数据形状 [6,57] -> [6,1,52]
        input = input.unsqueeze(1)
        # 把数据送给模型 提取事物特征
        # 数据形状 [seqlen,1,57],[1,1,128], [1,1,128]) -> [seqlen,1,128],[1,1,128],[1,1,128]
        rr, (hn, cn) = self.rnn(input, (hidden, c))
        # 数据形状 [seqlen,1,128] - [1, 128]
        tmprr = rr[-1]
        tmprr = self.linear(tmprr)
        return self.softmax(tmprr), hn, cn

    def inithidden(self):
        # 初始化隐藏层输入数据 inithidden()
        hidden = c = torch.zeros(size=(self.num_layers, 1, self.hidden_size))
        return hidden, c

def dm02_test_mylstm():
    # 1 实例化rnn对象
    my_lstm = LSTM(input_size=57, hidden_size=128, output_size=18)
    # 2 准备数据
    input = torch.randn(size=(6, 57))
    print(input.shape)
    hidden, c = my_lstm.inithidden()
    # 3 给模型1次性的送数据
    # [seqlen,57], [1,1,128]) -> [1,18], [1,1,128]
    output, hidden, c = my_lstm(input, hidden, c)
    print('一次性的送数据：output->', output.shape, output)
    print('hidden->', hidden.shape)
    # 4 给模型1个字符1个字符的喂数据
    hidden, c = my_lstm.inithidden()
    for i in range(input.shape[0]):
        tmpinput = input[i].unsqueeze(0)
        output, hidden, c = my_lstm(tmpinput, hidden, c)
    # 最后一次ouput
    print('一个字符一个字符的送数据output->', output.shape, output)

# GRU类 实现思路分析：
# 1 init函数 准备三个层 self.rnn self.linear self.softmax=nn.LogSoftmax(dim=-1)
    # def __init__(self, input_size, hidden_size, output_size, num_layers=1)
# 2 forward(input, hidden)函数
    # 让数据经过三个层 返回softmax结果和hn
    # 形状变化 [seqlen,1,57],[1,1,128]) -> [seqlen,1,128],[1,1,128]
# 3 初始化隐藏层输入数据 inithidden()
    # 形状[self.num_layers, 1, self.hidden_size]
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        # 1 init函数 准备三个层 self.rnn self.linear self.softmax=nn.LogSoftmax(dim=-1)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # 定义rnn层
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        # 定义linear层
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # 定义softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # 让数据经过三个层 返回softmax结果和hn
        # 数据形状 [6,57] -> [6,1,52]
        input = input.unsqueeze(1)
        # 把数据送给模型 提取事物特征
        # 数据形状 [seqlen,1,57],[1,1,128]) -> [seqlen,1,18],[1,1,128]
        rr, hn = self.rnn(input, hidden)
        # 数据形状 [seqlen,1,128] - [1, 128]
        tmprr = rr[-1]
        tmprr = self.linear(tmprr)
        return self.softmax(tmprr), hn

    def inithidden(self):
        # 初始化隐藏层输入数据 inithidden()
        return torch.zeros(self.num_layers, 1,self.hidden_size)

def dm_test_rnn_lstm_gru():
    # one-hot编码特征57（n_letters），也是RNN的输入尺寸
    input_size = n_letters
    # 定义隐层的最后一维尺寸大小
    n_hidden = 128
    # 输出尺寸为语言类别总数n_categories 1个字符预测成18个类别
    output_size = n_categories
    # 1 获取数据
    myfilename = './data/name_classfication.txt'
    my_list_x, my_list_y = read_data(myfilename)
    print('categories--->', categories)
    # 2 实例化dataset对象
    nameclassdataset = NameClassDataset(my_list_x, my_list_y)
    # 3 实例化dataloader
    mydataloader = DataLoader(dataset=nameclassdataset, batch_size=1, shuffle=True)
    my_rnn = RNN(input_size, n_hidden, output_size)
    my_lstm = LSTM(input_size, n_hidden, output_size)
    my_gru = GRU(input_size, n_hidden, output_size)
    print('rnn 模型', my_rnn)
    print('lstm 模型', my_lstm)
    print('gru 模型', my_gru)
    for i, (x, y) in enumerate(mydataloader):
        # print('x.shape', x.shape, x)
        # print('y.shape', y.shape, y)
        output, hidden = my_rnn(x[0], my_rnn.inithidden())
        print("rnn output.shape--->:", output.shape, output)
        break
    for i, (x, y) in enumerate(mydataloader):
        # print('x.shape', x.shape, x)
        # print('y.shape', y.shape, y)
        # 初始化一个三维的隐藏层0张量, 也是初始的细胞状态张量
        hidden, c = my_lstm.inithidden()
        output, hidden, c = my_lstm(x[0], hidden, c)
        print("lstm output.shape--->:", output.shape, output)
        break
    for i, (x, y) in enumerate(mydataloader):
        # print('x.shape', x.shape, x)
        # print('y.shape', y.shape, y)
        output, hidden = my_gru(x[0], my_gru.inithidden())
        print("gru output.shape--->:", output.shape, output)
        break


if __name__ == '__main__':

    # 读取数据
    my_list_x, my_list_y = read_data('data/name_classfication.txt')

    # 构造数据源
    dataset = NameClassDataset(my_list_x, my_list_y)
    print('dataset样本条数--->', len(dataset))
    print('dataset第0条样本--->', dataset[0])

    # 构造数据迭代器
    # dataloader = dm_test_NameClassDataset()

    # dm01_test_myrnn()

    # dm02_test_mylstm()

    # dm_test_rnn_lstm_gru()


