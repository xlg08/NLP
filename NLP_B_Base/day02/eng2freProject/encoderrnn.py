from preprocess import *


class EncoderRNN(nn.Module):
    # todo:1- 定义构造方法 init
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # 输入特征维度属性  input_size是英文词表的大小
        self.input_size = input_size
        # 词嵌入层和隐藏层特征维度属性  共用
        self.hidden_size = hidden_size
        # 词嵌入层对象属性
        self.embedding = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.hidden_size)
        # gru层对象属性
        # input_size: 上一层输出特征维度数
        # hidden_size: 当前层输出特征维度数
        # batch_first: x和hidden形状  ->  (句子数, 句子长度, 词维度)
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)

    # todo:2- 定义前向传播方法 forward
    def forward(self, input, hidden):

        # print('input--->', input.shape)
        # 词嵌入操作 词向量化
        embedded = self.embedding(input)  # 经过embedding升维升到了三维
        # print('embedded--->', embedded)     # 返回的是句子词向量矩阵
        # print('embedded.shape--->', embedded.shape)    # batch_size, seq_len, embedding_size
        # gru层前向传播操作
        output, hn = self.gru(embedded, hidden)
        # print('output.shape--->', output.shape)     # batch_size、句子数, seq_len、句子长度, hidden_size
        # print('hn.shape--->', hn.shape)      # num_layer, batch_size, hidden_size
        return output, hn

    # todo:3- 定义初始化隐藏状态值方法 inithidden
    def inithidden(self):
        return torch.zeros(size=(1, 1, self.hidden_size), device=device)


if __name__ == '__main__':

    # 获取数据
    (english_word2index, english_index2word, english_word_n,
     french_word2index, french_index2word, french_word_n, my_pairs) = my_getdata()

    # 创建张量数据集
    my_dataset = MyPairsDataset(my_pairs, english_word2index, french_word2index)

    # 创建数据加载器
    # batch_size: 当前设置为1, 因为句子长度不一致
    my_dataloader = DataLoader(dataset=my_dataset, batch_size=1, shuffle=True)
    # 创建编码器对象
    # input_size ：编码器输入特征的维度，即：英语单词词向量的维度，也是英语词表的大小
    my_encoderrnn = EncoderRNN(input_size=english_word_n, hidden_size=256).to(device=device)
    # x 返回的是每一个英语句子对应的单词的索引张量，即：每个英语样本句子的句子向量
    # y 返回的是每一个法语句子对应的单词的索引张量，即：每个法语样本句子的句子向量
    for i, (x, y) in enumerate(my_dataloader):

        # 一次性喂数据
        # 初始化隐藏状态值
        # print("x的形状是：", x.shape)  # batch_size, seq_len
        hidden = my_encoderrnn.inithidden()
        encoder_output, hn = my_encoderrnn(x, hidden)
        # print('encoder_output--->', encoder_output.shape)
        # print('hn--->', hn.shape)

        # # 一个时间步一个时间步喂数据, gru底层实现
        # hidden = my_encoderrnn.inithidden()
        # # x.shape[1]: 获取当前x的token数, 时间步数
        # for j in range(x.shape[1]):
        #     # print('x--->', x)
        #     # print('x[0]--->', x[0])
        #     # print('x[0][j]--->', x[0][j])
        #     # print('x[0][j].shape--->', x[0][j].shape)       # torch.Size([])
        #     tmp_x = x[0][j].view(1, -1)
        #     # print('tmp_x--->', tmp_x)
        #     # print('tmp_x.shape--->', tmp_x.shape)		# torch.Size([1, 1])
        #     output, hidden = my_encoderrnn(tmp_x, hidden)
        # # print('观察：最后一个时间步output输出是否相等')  # hidden_size = 8 效果比较好
        # # print('encoder_output[0][-1]===>', encoder_output[0][-1])
        # # print('output===>', output)
        break
