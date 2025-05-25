from torch import nn
import torch
from data_preprocess import n_letters, n_categories


# 构建模型
class RNN(nn.Module):
    # 初始化
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):

        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size

        # RNN 循环神经网络层
        #                 词向量维度    隐藏状态维数(隐藏层神经元个数)   隐藏层层数
        self.rnn = nn.RNN(input_size, hidden_size, num_layers)
        print('卷积层对象：', self.rnn)         # RNN(57, 64, num_layers=1)       # num_layers=1 时 num_layers 不显示

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
        x = x.unsqueeze(1)  # 在一维的位置，添加一个维度为1的新维度，在该位置就是 批次(一次输入几个单词)
        # 一次性
        print("输出特征的维度：", x.shape)

        # 模型训练
        # 卷积层
        out, hn = self.rnn(x, h0)
        # print(out[-1] == hn)      # 都是True  最后的输出 就是 hn

        # fc 全连接层
        print("out的形状：", out.shape)         # 一次性输入时：torch.Size([4, 1, 64])    一个单词一个单词的输入时：torch.Size([1, 1, 64]) 输入四次
        print("out[-1]的形状：", out[-1].shape)         # torch.Size([1, 64])
        # print("out：", out)
        out = self.fc(out[-1])  # 只需要最后的输出就可以，因为单词是很短的文本序列

        # 激活函数
        print(f"全连接层的输出：", out)
        out = self.softmax(out)

        return out, hn      # out 是 经过 softmax 激活之后的输入， hn 是卷积层最后的隐藏状态

    # 3 初始化隐藏层输入数据 inithidden()
    #   形状[self.num_layers, 1, self.hidden_size]
    def initHidden(self):
        #                隐藏层层数       批次数  隐藏层神经元数量（隐藏状态的维数）
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)

        return h0


if __name__ == '__main__':

    model = RNN(input_size=n_letters, hidden_size=64, output_size=n_categories)

    x = torch.randn(4, n_letters)  # 输入一个词，并且假设一个词由四个字母组成
    h0 = model.initHidden()

    # 单词一次性输入
    output, hn = model(x, h0)
    # print(output)
    # print(output.shape)
    # print(hn)
    # print(hn.shape)

    # 单词一个一个输入模型
    hidden = h0
    # print("x.shape：", x.shape)      # torch.Size([4, 57])  输入的词的 onehot 形状
    for i in range(x.shape[0]):
        # print(f'x[{i}]：{x[i]}')
        # print("转换前词的每个字母的形状：", x[i].shape)     # 每个词的形状是 torch.Size([57])
        # unsqueeze(0) 用于 在指定位置添加一个新的维度（维数为1）。
        input = x[i].unsqueeze(0)  # 表示的维度为 (1, 57)   就是这个单词的一个字母 用 57 维的向量表示
        # model的 forword() 前向传播时，接受的输入特征时二维的
        # print("转换后的词：", input)
        # print("转换后词的每个字母的形状：", input.shape)     # 每个词的形状是 torch.Size([1, 57])

        # output, hidden = model(input, hidden)
        # print(output)
