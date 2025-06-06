import torch
import torch.nn as nn


# 创建神经网络类
class Attn(nn.Module):

    # todo:1-初始化init构造函数
    def __init__(self, query_size, key_size, value_size1, value_size2, output_size):
        super().__init__()

        # 1-1: 初始化对象属性
        self.query_size = query_size  # 解码器隐藏状态值的特征维度 或 输入x的特征维度
        self.key_size = key_size  # 编码器隐藏状态值的特征维度 解码器第一个隐藏状态=编码器最后一个隐藏状态
        # value-> 编码器的output结果 (batch_size, seq_len, hidden_size)
        # self.value_size1 = value_size1  # seq_len, 句子长度, 时间步数量, token数
        self.seq_len = value_size1  # seq_len, 句子长度, 时间步数量, token数
        # self.value_size2 = value_size2  # hidden_size, gru层提取的特征维度
        self.hidden_size = value_size2  # hidden_size, gru层提取的特征维度
        self.output_size = output_size  # 解码器输入的特征维度

        # 1-2: 定义线性层1 -> q和k在特征维度轴拼接后进行线性计算
        # 输入特征数/输入神经元个数: self.query_size + self.key_size q和k在特征维度轴拼接
        # 输出特征数/输出神经元个数: self.value_size1
        # 权重概率矩阵 * value = (b, n, m) * (b, m, p) = (b, n, p)  (b,n,p)代表是动态c的形状
        # value->(b, m, p) b=1, m=value_size1 p=value_size2
        # 权重概率矩阵的形状是由 attn线性层决定 (b, n, m) -> m=value_size1
        self.attn = nn.Linear(self.query_size + self.key_size, self.seq_len)

        # 1-3: 定义线性层2 -> q和动态c在特征维度轴拼接后进行线性计算, 得到当前时间步的输入X(X=w(q+c))
        # self.query_size + self.value_size2->为什么加value_size2?
        # 	q和动态c在特征维度上拼接,而c形状 -> (b, m, p) 	p = value_size2
        self.attn_combine = nn.Linear(self.query_size + self.hidden_size, self.output_size)

    # todo:2- 定义forward方法
    def forward(self, Q, K, V):
        # print('Q--->', Q.shape, Q)
        # print('K--->', K.shape, K)
        # print('V--->', V.shape, V)

        # 2-1 拼接 + 线性计算 + softmax -> 权重概率矩阵
        # print('Q[0]--->', Q[0].shape, Q[0])  #  形状 -> (句子长度, 词向量维度)
        # dim=1: 特征维度进行拼接
        # print('q和k拼接结果--->', torch.cat(tensors=[Q[0], K[0]], dim=1).shape)
        # print('q和k拼接后线性计算结果--->', self.attn(torch.cat(tensors=[Q[0], K[0]], dim=1)).shape)
        # dim=-1:按行进行softmax计算
        # Q.shape :
        attn_weights = torch.softmax(self.attn(torch.cat(tensors=[Q[0], K[0]], dim=1)), dim=-1)
        print('权重概率矩阵--->', attn_weights.shape, attn_weights)       # torch.Size([1, 32])

        # 2-2 权重概率矩阵和 V 矩阵相乘 -> 动态张量c
        # attn_weights.unsqueeze(0) -> 将二维转换成三维 (句子数, 句子长度, 词向量维度)
        attn_applied = torch.bmm(input=attn_weights.unsqueeze(0), mat2=V)
        print('动态张量c--->', attn_applied.shape, attn_applied)

        # 2-3 Q和动态张量c融合 -> 得到 解码器当前时间步的输入X, 后续将X输入到gru层 nn.GRU(X, h0)
        q_c_cat = torch.cat(tensors=[Q[0], attn_applied[0]], dim=1)
        print('q和动态张量c融合结果--->', q_c_cat.shape, q_c_cat)
        # gru层的输入X是三维, 所以进行 unsqueeze(0)
        output = self.attn_combine(q_c_cat).unsqueeze(0)
        print('gru当前时间步的输入X(q和c融合结果)--->', output.shape, output)
        # outpu后续输入到gru层
        # 伪代码: output2, hn = nn.GRU(output, h0)

        return output, attn_weights


if __name__ == '__main__':
    # 定义特征维度
    query_size = 32
    key_size = 32
    value_size1 = 32
    value_size2 = 64
    output_size = 32

    # 创建测试q,k,v
    # 形状 -> (batch_size, seq_len, embedding)
    Q = torch.randn(1, 1, query_size)
    K = torch.randn(1, 1, key_size)
    V = torch.randn(1, value_size1, value_size2)

    # 创建实例对象
    my_attn = Attn(query_size, key_size, value_size1, value_size2, output_size)

    # 调用实例对象, 自动执行forward方法
    output, attn_weights = my_attn(Q, K, V)
    print('=' * 80)
    print('输出结果--->', output.shape, output)
    print('权重概率矩阵--->', attn_weights.shape, attn_weights)
