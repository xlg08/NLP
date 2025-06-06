from torch import nn
import torch
from data_preprocess import n_letters, n_categories, read_data, NameClassDataset
from torch.utils.data import DataLoader


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 构建模型
class RNN(nn.Module):
    # 初始化
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, batch_size=1):

        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

        # RNN 循环神经网络层
        #                 词向量维度    隐藏状态维数(隐藏层神经元个数)   隐藏层层数
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        # print('卷积层对象：', self.rnn)         # RNN(57, 64, num_layers=1)       # num_layers=1 时 num_layers 不显示

        # fc    全连接层
        #                 隐藏层神经元个数  全连接层神经元个数
        self.fc = nn.Linear(hidden_size, output_size)

        # 定义softmax层
        # LogSoftmax + NLLLoss = CrossEntropyLoss
        # 后续多分类交叉熵损失函数使用NLLLoss
        # nn.LogSoftmax 是 PyTorch 中用于实现 Log-Softmax 函数 的一个模块，定义在 torch.nn 中。
        # LogSoftmax = log(softmax(x))
        #   常用于多分类任务的输出层，尤其是在与 nn.NLLLoss（负对数似然损失）结合使用时。
        # dim：指定在哪个维度上应用 Softmax。
        #   dim=-1 即最后一维，也就是张量最内部的元素做 LogSoftmax。
        #   如果不确定数据的形状，用 dim=-1 是一个更通用和安全的做法。
        self.softmax = nn.LogSoftmax(dim=-1)

    # 前向传播
    # 2 forward(input, hidden)函数
    #   让数据经过三个层 返回softmax结果和hn
    #   形状变化 [seq_len,1,57],[1,1,64]) -> [seq_len,1,64],[1,1,64]
    def forward(self, x, h0):
        # batch
        # x = x.unsqueeze(1)  # 在一维的位置，添加一个维度为1的新维度，在该位置就是 批次(一次输入几个单词)
        x = x.transpose(0, 1)  # 转换维度，batch_size 和 seq_len
        # 一次性

        # print("输出特征的维度：", x.shape)

        # 模型训练
        # 卷积层
        out, hn = self.rnn(x, h0)
        # print(out[-1] == hn)      # 都是True  最后的输出 就是 hn

        # fc 全连接层
        # 一次性输入时：torch.Size([4, 1, 64])    一个单词一个单词的输入时：torch.Size([1, 1, 64]) 输入四次
        # print("out的形状：", out.shape)
        # print("out[-1]的形状：", out[-1].shape)         # torch.Size([1, 64])
        # print("卷积层的out输出为：", out)
        out = self.fc(out[-1])  # 只需要最后的输出就可以，因为单词是很短的文本序列

        # print(f"全连接层的out输出：", out)
        # 激活函数
        out = self.softmax(out)

        return out, hn  # out 是 经过 softmax 激活之后的输入， hn 是卷积层最后的隐藏状态

    # 3 初始化隐藏层输入数据 inithidden()
    #   形状[self.num_layers, 1, self.hidden_size]
    def initHidden(self, x=None):
        #                隐藏层层数       批次数  隐藏层神经元数量（隐藏状态的维数）
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        # h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        # h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        # h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        # h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=device)
        # print("h0的形状为：", h0.shape)      # h0的形状为：torch.Size([1, 1, 64])
        return h0


# lstm 模型 Long Short-Term Memory
class LSTM(nn.Module):
    # 属性初始化+层初始化
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, batch_size=1):
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
        # self.batch_size = batch_size

        # 实例化lstm
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        # 全连接层 fc
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h0, c0):
        # 扩展batch维
        # x = x.unsqueeze(1)
        x = x.transpose(0, 1)
        # LSTM
        out, (hn, cn) = self.rnn(x, (h0, c0))
        # 全连接层
        out = self.fc(out[-1])
        # softmax
        out = self.softmax(out)
        # 返回输出
        return out, hn, cn

    # 初始化隐藏状态和细胞状态
    def initHidden(self, x):
        # 全零
        # h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        # h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=device)
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)

        c0 = h0
        return h0, c0


class GRU(nn.Module):
    # 属性初始化+层初始化
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
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
        # 实例化GRU
        self.rnn = nn.GRU(self.input_size, self.hidden_size, self.num_layers)
        # 全连接层 fc
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # softmax
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, h0):
        # 扩展batch维
        # x = x.unsqueeze(1)
        x = x.transpose(0, 1)
        # GRU
        out, hn = self.rnn(x, h0)
        # 全连接层
        out = self.fc(out[-1])
        # softmax
        out = self.softmax(out)
        # 返回输出
        return out, hn

    # 初始化隐藏状态和细胞状态
    def initHidden(self, x):
        # 全零
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        # h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=device)
        return h0


def tes_model():
    '''
    测试模型能力 RNN、LSTM、GRU
    :return:
    '''
    # 1.设置超参数
    batch_size = 4
    hidden_size = 100
    input_size = n_letters
    output_size = n_categories

    # 2.读取文件
    x_list, y_list = read_data("data/name_classfication.txt")

    # 3.构建dataset
    dataset = NameClassDataset(x_list, y_list, n_letters)

    # 4.dataloader 获取数据迭代
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # 5.模型实例化
    rnn = RNN(input_size, hidden_size, output_size)
    lstm = LSTM(input_size, hidden_size, output_size)
    gru = GRU(input_size, hidden_size, output_size)

    # 6.数据送入模型中进行测试
    # 6.1 测rnn
    # 遍历数据
    for x, y in dataloader:
        print(f"x：{x}, y：{y}")
        # DataLoader 返回的形状是 (批次，seq_len，input_size)
        # x.shape：torch.Size([1, 8, 57]), y.shape：torch.Size([1])
        # x.shape：torch.Size([1, 19, 57]), y.shape：torch.Size([1])
        print(f"x.shape：{x.shape}, y.shape：{y.shape}")
        print(f"x[0].shape：{x[0].shape}")  # x[0].shape：torch.Size([8, 57])

        # 初始化隐藏状态
        h0 = rnn.initHidden()
        # 模型推理
        output, hn = rnn(x[0], h0)
        # output, hn = rnn(x, h0)               # 以批次传入数据
        # 输出展示
        print(output)
        print(output.shape)
        break

    # 6.2 测LSTM
    # for x, y in dataloader:
    #     # 初始化隐藏状态
    #     h0, c0 = lstm.initHidden()
    #     # 模型推理
    #     output, hn, cn = lstm(x[0], h0, c0)
    #     # 输出展示
    #     print(output)
    #     print(output.shape)
    #     break

    # 6.3 测gru
    # 遍历数据
    # for x, y in dataloader:
    #     # 初始化隐藏状态
    #     h0 = gru.initHidden()
    #     # 模型推理
    #     output, hn = gru(x[0], h0)
    #     # 输出展示
    #     print(output)
    #     print(output.shape)
    #     break


if __name__ == '__main__':
    batch_size=30
    # 模型实例化
    model = RNN(input_size=n_letters, hidden_size=64, output_size=n_categories,batch_size=batch_size)

    # model = LSTM(input_size=n_letters, hidden_size=64, output_size=n_categories)

    # model = GRU(input_size=n_letters, hidden_size=64, output_size=n_categories)

    # 输入数据[seq_len,input_size]
    x = torch.randn(4, 30,n_letters)  # 输入一个词，并且假设一个词由四个字母组成

    # 初始化隐藏状态
    # RNN GRU
    h0 = model.initHidden()
    # LSTM
    # h0, c0 = model.initHidden()

    # 模型推理
    # RNN GRU
    # 单词一次性输入
    output, hn = model(x, h0)
    # LSTM
    # output, hn, cn = model(x, h0, c0)

    # 结果展示
    # 【bs,num_class】
    print(output)
    print(output.shape)
    # [num_layer,bs,hs]
    # print(hn)
    # print(hn.shape)

    # 单词一个一个输入模型，单个元素送入网络中，结果
    # hidden = h0
    # print("x.shape：", x.shape)      # torch.Size([4, 57])  输入的词的 onehot 形状
    # LSTM
    # cell = c0
    # 遍历数据
    # for i in range(x.shape[0]):
    # print(f'x[{i}]：{x[i]}')
    # print("转换前词的每个字母的形状：", x[i].shape)     # 每个词的形状是 torch.Size([57])
    # unsqueeze(0) 用于 在指定位置添加一个新的维度（维数为1）。
    # 扩展seq_len
    # input = x[i].unsqueeze(0)  # 表示的维度为 (1, 57)   就是这个单词的一个字母 用 57 维的向量表示
    # model的 forword() 前向传播时，接受的输入特征时二维的
    # print("转换后的词：", input)
    # print("转换后词的每个字母的形状：", input.shape)     # 每个词的形状是 torch.Size([1, 57])

    # 模型处理
    # RNN GRU
    # output, hidden = model(input, hidden)
    # LSTM
    # output, hidden, cell = model(input, hidden, cell)
    # 每个元素的输出结果
    # print(output)

    # tes_model()
