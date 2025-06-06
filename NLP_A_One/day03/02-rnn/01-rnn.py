import torch.nn as nn
import torch

# 1.实例化RNN
# input_size 输入数据的维数，也就是词嵌入的维度
# hidden_size 隐藏状态维数
# num_layers 隐藏层的层数
# batch_first ：是否把 输入参数的 batch 放在最前面，True为放到最前面，False为放到中间， batch 放在中间是最好的
# 当 batch_first设置为True时，
#   输入的参数顺序变为：x：[batch, seq_len, input_size]，h0：[batch, num_layers, hidden_size]。
rnn = nn.RNN(input_size=5, hidden_size=4, num_layers=2, batch_first=False)

# 批次大小：并行处理，一次处理几个样本

# 2.输入数据x
# 句子长度，批次大小（句子个数），词嵌入维数
x = torch.randn(10, 3, 5)
print("输入数据：", x)

# 3.隐藏状态数据
# 隐藏层层数，批次大小（句子个数），隐藏状态维数
h0 = torch.randn(2, 3, 4)
print(f"初始化隐藏状态：{h0}")

# 4.获取输出
output, ht = rnn(x, h0)

# 5.打印输出
print(output)
print(output.shape)
print(ht)
print(ht.shape)


def 输出权重():
    # print("*" * 50)
    # print("所有权重存到一个列表中：", type(rnn.all_weights))      # 返回 list 类型
    # print("所有权重：", len(rnn.all_weights))            # 隐藏层有几层则 all_weights 是长度为几的列表 num_layers=1 时，则有1个列表
    # print("所有权重：", rnn.all_weights)

    # print("*"*50)
    # # print("所有权重存到一个列表中：", type(rnn.all_weights[0]))       # 返回 list 类型
    # print("所有权重：", len(rnn.all_weights[0]))         # 4个
    # print("所有权重：", rnn.all_weights[0])         #
    #
    # print("*"*50)
    # # print("所有权重存到一个列表中：", type(rnn.all_weights[0][0]))      # 返回 <class 'torch.nn.parameter.Parameter'>
    # # print("所有权重：", len(rnn.all_weights[0][0]))      # 4个 等于 hidden_size 隐藏层的个数
    # print("所有权重：", rnn.all_weights[0][0])
    #
    # print("*"*50)
    # # print("所有权重存到一个列表中：", type(rnn.all_weights[0][0][0]))       # 返回 <class 'torch.Tensor'>
    # print("所有权重：", len(rnn.all_weights[0][0][0]))       # 5个 等于 input_size 输入数据的维数，也就是词嵌入的维度的个数
    # print("所有权重：", rnn.all_weights[0][0][0])

    print(rnn.state_dict)


# 输出权重()



