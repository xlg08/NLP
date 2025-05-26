from torch import nn
import torch
from data_preprocess import n_letters, n_categories, read_data, NameClassDataset
from torch.utils.data import DataLoader


# 构建模型
class RNN(nn.Module):
    # 初始化
    def __init__(self, input_size, hidden_size, output_size, batch_size=1,num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        # RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # fc
        self.fc = nn.Linear(hidden_size, output_size)
        # softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h0):
        # batch
        # x = x.unsqueeze(1)
        x = torch.transpose(x, 0, 1)
        # 模型训练
        out, hn = self.rnn(x, h0)
        # fc
        out = self.fc(out[-1])
        # 激活
        out = self.softmax(out)
        return out, hn

    def initHidden(self):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return h0


class LSTM(nn.Module):
    # 属性初始化+层初始化
    def __init__(self, input_size, hidden_size, output_size, batch_size=1,num_layers=1):
        '''
        :param input_size: 词嵌入维数
        :param hidden_size: 隐藏状态和细胞状态的维数
        :param output_size: 分类个数
        :param num_layers: LSTM中层数
        '''
        # 父类的init
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        # 实例化lstm
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        # 全连接层 fc
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h0, c0):
        # 扩展batch维
        # x = x.unsqueeze(1)
        x =torch.transpose(x,0,1)
        # LSTM
        out, (hn, cn) = self.rnn(x, (h0, c0))
        # 全连接层
        out = self.fc(out[-1])
        # softmax
        out = self.softmax(out)
        # 返回输出
        return out, hn, cn

    # 初始化隐藏状态和细胞状态
    def initHidden(self):
        # 全零
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        c0 = h0
        return h0, c0


class GRU(nn.Module):
    # 属性初始化+层初始化
    def __init__(self, input_size, hidden_size, output_size,batch_size=1, num_layers=1):
        '''
        :param input_size: 词嵌入维数
        :param hidden_size: 隐藏状态和细胞状态的维数
        :param output_size: 分类个数
        :param num_layers: LSTM中层数
        '''
        # 父类的init
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        # 实例化GRU
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        # 全连接层 fc
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h0):
        # 扩展batch维
        # x = x.unsqueeze(1)
        x =torch.transpose(x,0,1)
        # GRU
        out, hn = self.rnn(x, h0)
        # 全连接层
        out = self.fc(out[-1])
        # softmax
        out = self.softmax(out)
        # 返回输出
        return out, hn

    # 初始化隐藏状态和细胞状态
    def initHidden(self):
        # 全零
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return h0


def test_model():
    '''
    测试模型能力RNN，LSTM，GRU
    :return:
    '''
    # 1.设置超参数
    batch_size = 4
    hidden_size = 100
    input_size = n_letters
    output_size = n_categories
    # 2.读取文件
    x_list, y_list = read_data('data/name_classfication.txt')
    # 3.构建dataset
    dataset = NameClassDataset(x_list, y_list,n_letters)
    # 4.dataloader获取数据迭代
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    # 5、模型实例化 RNN，LSTM，GRU
    rnn = RNN(input_size, hidden_size, output_size,batch_size=batch_size,num_layers=1)
    lstm = LSTM(input_size, hidden_size, output_size,batch_size=batch_size,num_layers=1)
    gru = GRU(input_size, hidden_size, output_size,batch_size=batch_size,num_layers=1)
    # 6.数据送入模型中进行测试
    # 6.1 测rnn
    # 遍历数据
    for x, y in dataloader:
        # 初始化隐藏状态
        h0 = rnn.initHidden()
        # 模型推理
        # output, hn = rnn(x[0], h0)
        output, hn = rnn(x, h0)
        #输出展示
        print(output)
        print(output.shape)
        break
    # 6.2 测LSTM
    for x, y in dataloader:
        # 初始化隐藏状态
        h0,c0 = lstm.initHidden()
        # 模型推理
        output, hn,cn = lstm(x, h0,c0)
        # output, hn,cn = lstm(x[0], h0,c0)
        #输出展示
        print(output)
        print(output.shape)
        break

    # 6.3 测gru
    # 遍历数据
    for x, y in dataloader:
        # 初始化隐藏状态
        h0 = gru.initHidden()
        # 模型推理
        # output, hn = gru(x[0], h0)
        output, hn = gru(x, h0)
        # 输出展示
        print(output)
        print(output.shape)
        break


if __name__ == '__main__':
    # # 模型实例化
    # # model = RNN(input_size=n_letters, hidden_size=64, output_size=n_categories)
    # # model = LSTM(input_size=n_letters, hidden_size=64, output_size=n_categories)
    # model = GRU(input_size=n_letters, hidden_size=64, output_size=n_categories)
    # # 输入数据[seq_len,input_size]
    # x = torch.randn(4, n_letters)
    # # 初始化隐藏状态
    # # RNN GRU
    # h0 = model.initHidden()
    # # LSTM
    # # h0, c0 = model.initHidden()
    # # 模型推理
    # # RNN GRU
    # output, hn = model(x, h0)
    # # LSTM
    # # output, hn, cn = model(x, h0, c0)
    # # 结果展示
    # # 【bs,num_class】
    # print(output)
    # print(output.shape)
    # # [num_layer,bs,hs]
    # print(hn)
    # print(hn.shape)
    # # 单个元素送入网络中，结果
    # hidden = h0
    # # LSTM
    # # cell = c0
    # # 遍历数据
    # for i in range(x.shape[0]):
    #     # print(x[i])
    #     # print(x[i].shape)
    #     # 扩展seq_len
    #     input = x[i].unsqueeze(0)
    #     # 模型处理
    #     # RNN GRU
    #     output, hidden = model(input, hidden)
    #     # LSTM
    #     # output, hidden, cell = model(input, hidden, cell)
    #     # 每个元素的输出结果
    #     print(output)
    test_model()
