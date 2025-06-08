import copy

from input import *


# 定义缩放点积注意力规则函数, 方便后续调用
def attention(query, key, value, mask=None, dropout=None):
    """
    注意力计算封装函数
    :param query:   输入x  解码器掩码多头自注意力子层输出
    :param key:     输入x  编码器输出结果
    :param value:   输入x  编码器输出结果
    :param mask:    是否掩码
    :param dropout: dropout层对象 函数名
    :return:    动态c, 权重概率矩阵
    """

    # todo:1- 获取d_k, 词维度数
    d_k = query.shape[-1]  # query 的形状是 句子数, 句子长度, 词向量维数
    # print('d_k--->', d_k)			# 512 维

    # todo:2- q和k计算权重分数矩阵
    # 多头注意力机制中， 传入的 Q, K, V 都是四维的，因此只能使用 matmul() 矩阵乘法方法， bmm() 方法只适用于三维矩阵的乘法，而且 matmul()方法支持广播
    # 四维的矩阵乘法就是 (b, h, n, p) * (b, h, p, m) = (b, h, n, m)
    # 需要满足 做乘法的两个矩阵的前两个维度的维数是相同的，以及第一个矩阵的最后一个维度的维数与第二个矩阵的第三个维度的维数是相同的，才能进行矩阵乘法
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    print('scores --->', scores)
    # q的形状 句子数，q的句子长度，q的词向量维度
    # k的形状 句子数，k的句子长度，k的词向量维度
    # k转置的形状 句子数, k的词向量维度，k的句子长度
    # 句子数, Q的句子长度数, K的句子长度 (相关性是通过Q、K计算出来的)
    # print('scores.shape --->', scores.shape)

    # todo:3- 判断是否需要进行掩码操作
    if mask is not None:
        # print(mask == 0)
        # (mask == 0).shape： torch.Size([2, 1, 1, 4])         # 四维下：句子数，头数，Q句子长度，K句子长度
        # print("(mask == 0).shape：", (mask == 0).shape)
        scores = scores.masked_fill(mask == 0, value=-1e9)
    # print('='*50)
    # print('scores --->', scores)
    # print('scores.shape --->', scores.shape)

    # todo:4- 权重分数矩阵进行softmax操作, 得到权重概率矩阵
    p_attn = torch.softmax(scores, dim=-1)
    # print('p_attn--->', p_attn.shape, p_attn)

    # todo:5- 判断是否对权重概率矩阵进行dropout正则化
    if dropout is not None:
        p_attn = dropout(p_attn)

    # todo:6- 计算动态c矩阵
    c = torch.matmul(p_attn, value)  # value 的 形状是 句子数, V的句子长度, 词向量维数
    # print('动态 c --->', c)
    # print('动态 c.shape --->', c.shape)  # 句子数，句子长度，词向量维数

    return c, p_attn  # 动态张量C, 权重概率矩阵


# 定义克隆函数, 用于克隆不同子层（但是所有克隆出的子层）
def clones(module, N):
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))


# 创建多头注意力机制类
class MultiHeadedAttention(nn.Module):

    # todo:1- init方法
    def __init__(self, head, d_model, dropout_p=0.1):
        super().__init__()

        assert d_model % head == 0, 'd_model不能被head整数'
        self.d_k = d_model // head  # 每一个头的注意力词向量维度
        self.dropout = nn.Dropout(p=dropout_p)
        self.head = head
        # 初始为None, 还没有计算注意力
        self.attn = None

        # 4个线性层
        # 前 3 个分别对 q, k, v 进行线性学习
        # 最后一个对多头注意力拼接结果进行线性学习
        self.linears = clones(nn.Linear(d_model, d_model), 4)

    # print('self.linears--->', self.linears)
    # self.linears---> ModuleList(
    #   (0-3): 4 x Linear(in_features=512, out_features=512, bias=True)
    # )
    # print('self.linears 的类型 --->', type(self.linears))
    # <class 'torch.nn.modules.container.ModuleList'>

    # todo:2- forward方法
    def forward(self, query, key, value, mask=None):

        # todo:1- 获取batch_size大小
        batch_size = query.size()[0]  # 获取句子数
        # print('batch_size--->', batch_size)

        # todo:2- 准备空列表, 存储线性计算 + 变形结果
        output_list = []        # 保存 线性计算并且变形为四维后的 Q、K、V

        # todo:3- q, k, v 分别进行线性计算
        for model, x in zip(self.linears, (query, key, value)):
            # print('model--->', model)		# 线性模型
            # print('x--->', x)
            output = model(x)  # 对 Q, K, V 进行线性计算
            print("output.shape：", output.shape)  # output.shape： torch.Size([2, 4, 512])

            # todo:4- 线性计算结果变形 -> (batch_size, seq_len, head, d_k)  根据多头注意力
            # transpose(1, 2): 头数和词数换位是为了，词数和词向量相邻, 更好的学习特征
            output = output.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            # 没将头数和词数换位之前的维度是 torch.Size([2, 4, 8, 64])
            print("output.shape：", output.shape)  # output.shape： torch.Size([2, 8, 4, 64])

            # todo:5- 将变形结果保存到空列表中
            output_list.append(output)

        # 获取q, k, v
        print('output_list--->', output_list)
        # 每个形状都是  torch.Size([2, 8, 4, 64])   		# 句子数, 头数, 句子长度(词数), 词向量维度
        print('output_list[0].shape--->', output_list[0].shape)
        print('len(output_list)--->', len(output_list))  # 三个
        # 分出 q, k, v
        query = output_list[0]
        # print("query", query)
        # # torch.Size([2, 8, 4, 64])			batch_size, 头数, seq_len, 词向量维度
        print("111query.shape：", query.shape)
        key = output_list[1]
        # print("key", key)
        value = output_list[2]
        # print("value", value)

        # 第一次 都是 True
        print("query == key ：", query == key)
        print("query == value ：", query == value)

        # todo:6- 计算多头注意力, 调用attention函数  (batch_size, seq_len, head, d_k)
        x, p_attn = attention(query, key, value, mask)
        # torch.Size([2, 8, 4, 64])   句子数, 头数, 句子长度, 词向量维数
        print('多头动态向量x.shape--->', x.shape)

        # todo:7- 多头注意力结果变形 -> (batch_size, seq_len, word_dim)
        # 在 PyTorch 中，contiguous() 是一个用于 Tensor 的方法，它的作用是 返回内存连续（contiguous）存储的张量副本。
        # 	这个方法经常出现在涉及 view()、reshape() 等操作之前，确保张量在内存中的布局是连续的。
        # x 现在的形状是 (句子数, 头数, 句子长度, 词向量维数)   更换 1、2 维 ，后形状为(句子数, 句子长度, 头数, 词向量维数)
        # 再改成 三维形状 (句子数, 句子长度, 头数*词向量维度 )   生成 (句子数, 句子长度, 新词向量维度 )
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        # print('x--->', x)
        # print('x.shape--->', x.shape)		# torch.Size([2, 4, 512])

        # todo:8- 经过线性层计算返回输出结果
        # self.linears[-1]: 线性层对象
        x = self.linears[-1](x)

        return x


# 前馈全连接层
# 在更高维的空间中进行更丰富的特征学习，提取更高级的特征。
# 并且引入了非线性能力，使其可以学习非线性关系，对每个位置的表示进行非线性转换，以增强模型的表达能力。
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p=0.1):
        super().__init__()

        # 定义两层线性层
        # d_ff > d_model
        self.linear1 = nn.Linear(d_model, d_ff)  # 升维，提取更丰富的特征，提高表达能力
        self.linear2 = nn.Linear(d_ff, d_model)  # 再将维度降回到原始的词向量维度，但是是拥有更丰富语义信息的词向量

        # 定义dropout层
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        output = torch.relu(self.linear1(x))  # 更高维度+非线性有助于模型更好地学习复杂映射。
        output = self.dropout(output)
        print('output --->', output)
        print('output.shape --->', output.shape)  # torch.Size([2, 4, 2048]) 前馈神经网络第一个线性层先升维到了2048
        return self.linear2(output)


# 层归一化层
class LayerNorm(nn.Module):
    # todo:1- init方法
    def __init__(self, features, eps=1e-6):

        super().__init__()
        self.eps = eps
        # 初始化w,全1, w维度数和features维度数一致
        self.w = nn.Parameter(torch.ones(features))
        # print('self.w--->', self.w.shape)
        # 初始化b,全0, b维度数和features维度数一致
        self.b = nn.Parameter(torch.zeros(features))

    # todo:2- forward方法
    def forward(self, x):
        # 计算x的均值和标准

        # 计算均值 因为是层归一化 （根据每个词的词向量）
        # 本质上就是 每个词 512 个词向量相加，再除以 512  因为一共有两句话每句话四个词，因此返回的是 (2, 4)
        # mean = x.mean(dim=-1)
        # keepdims: 默认False, 返回二维张量; True, 和原x维度数一致
        mean = x.mean(dim=-1, keepdims=True)  # 计算均值
        # print('mean --->', mean)
        # print('mean.shape --->', mean.shape, mean)

        std = x.std(dim=-1, keepdims=True)  # 计算方差
        # print('mean --->', mean)
        # print('mean.shape --->', mean.shape)

        # 计算标准化的结果
        x = self.w * ((x - mean) / (std + self.eps)) + self.b
        return x


# 子层连接(残差连接)
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout_p=0.1):
        super().__init__()
        self.size = size  # 主要用于层归一化中，初始化 K 和 B 时设置维度
        # 创建层归一化对象
        self.norm = LayerNorm(self.size)  # 在残差连接之后进行层归一化
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, sublayer):
        """
        计算子层结果  当前层输入x+dropout(当前层输出)=残差结果->norm
        :param x: 当前层输入
        :param sublayer: 当前层对象 方法名/内存地址
        :return: 子层结果
        """
        # 输入的 x 是经过词嵌入层和位置编码之后的结果
        # sublayer(x) 方法调用的是多头自注意力子层，返回的是 动态语义张量C
        # norm() 方法 是 层归一化层
        x = self.norm(x + self.dropout(sublayer(x)))  # 残差连接之后进行层归一化
        return x


# 创建编码器层
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forwad, dropout_p=0.1):
        super().__init__()
        self.size = size  # 词向量维度
        self.self_attn = self_attn  # 多头自注意力机制对象
        self.feed_forwad = feed_forwad  # 前馈全连接对象
        self.dropout_p = dropout_p

        # 克隆两个子层对象  (克隆子层连接的神经网络层)
        # self.sublayer->[对象1, 对象2]
        self.sublayer = clones(SublayerConnection(self.size, self.dropout_p), 2)

    # print('self.sublayer--->', self.sublayer)

    def forward(self, x, mask):
        """
        编码器层进行编码
        :param x: 上一层的输出, 如果是第一层编码器层,x是输入部分的输出
        :param mask: 填充掩码
        :return: 编码结果,提取到的新特征向量
        """
        # 多头自主力机制	子层连接计算
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        # 前馈全连接	子层连接计算
        x = self.sublayer[1](x, self.feed_forwad)

        return x


# 创建编码器，编码器由六层编码器组成
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)

    # print('self.layers--->', self.layers)
    # self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # 循环遍历编码器, 一层一层计算
        for layer in self.layers:
            x = layer(x, mask)

        return x


def te01_encoder():
    vocab = 1000  # 词表大小是1000
    d_model = 512  # 词嵌入维度是512维

    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 0], [491, 998, 0, 0]])

    # 输入部分的Embeddings类
    my_embeddings = Embeddings(vocab, d_model)
    embedded_result = my_embeddings(x)

    dropout_p = 0.1  # 置0概率为0.1
    max_len = 60  # 句子最大长度

    # 输入部分的PositionalEncoding类
    my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    pe_result = my_pe(embedded_result)

    # 调用attention函数
    # 准备q,k,v
    query = key = value = pe_result  # 自注意力
    # print('query--->', query.shape)
    # 准备mask掩码张量 padding_mask
    # unsqueeze(1) -> 形状(2,1,4) 后续进行masked_fill操作, 会进行广播, 变成 (2,4,4)
    # 多头注意力机制,需要得到(2,1,1,4)形状mask
    mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    # casual mask			# 因果掩码
    # mask = torch.tril(torch.ones(4, 4))		# 下三角矩阵
    # print('mask--->', mask.shape, mask)
    # c, p_attn = attention(query, key, value, mask)
    # print('c--->', c.shape, c)
    # print('p_attn--->', p_attn.shape, p_attn)

    head = 8
    # 创建多头注意力机制类对象
    my_mha = MultiHeadedAttention(head, d_model)
    # 实例化前馈全连接层对象
    d_ff = 2048
    my_ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    # 创建深拷贝对象
    c = copy.deepcopy
    # 创建编码器层对象
    # 深拷贝对象, 每个对象的内存地址不同, 不共享参数
    my_el = EncoderLayer(d_model, c(my_mha), c(my_ff), dropout_p)

    # 创建编码器对象
    my_encoder = Encoder(my_el, 2)
    encoder_result = my_encoder(pe_result, mask)
    # print('encoder_result--->', encoder_result.shape)
    return encoder_result


def te02_encoder():
    vocab = 1000  # 词表大小是1000
    d_model = 512  # 词嵌入维度是512维

    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 0], [491, 998, 0, 0]])

    # 输入部分的Embeddings类
    my_embeddings = Embeddings(vocab, d_model)
    embedded_result = my_embeddings(x)

    dropout_p = 0.1  # 置0概率为0.1
    max_len = 60  # 句子最大长度

    # 输入部分的PositionalEncoding类
    my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    pe_result = my_pe(embedded_result)

    # 调用attention函数
    # 准备q,k,v
    # query = key = value = pe_result  # 自注意力
    # print('query--->', query.shape)

    # 准备mask掩码张量 padding_mask
    # unsqueeze(1) -> 形状(2,1,4) 后续进行masked_fill操作, 会进行广播, 变成 (2,4,4)
    # print((x != 0).type(torch.uint8))		# 将张量中 True 和 False 的值，转化为 1 和 0 的uint8类型
    mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)
    # casual mask
    # mask = torch.tril(torch.ones(4, 4))
    # print('mask--->', mask.shape, mask)
    # c, p_attn = attention(query, key, value, mask)
    # print('c--->', c.shape, c)
    # print('p_attn--->', p_attn.shape, p_attn)

    head = 8
    # 创建多头注意力机制类对象
    my_mha = MultiHeadedAttention(head, d_model)
    # # 调用多头注意力机制对象
    # # mha_result = my_mha(query, key, value, mask)
    # # print('mha_result--->', mha_result.shape, mha_result)
    # # 定义匿名函数
    # """
    # def sublayer(x):
    #     return my_mha(x, x, x)
    # """
    # # 变量名 = lambda 参数: 参数的表达式
    # sublayer = lambda x: my_mha(x, x, x)
    # print('sublayer--->', sublayer)
    #
    # # 创建子层连接对象
    # size = d_model			#
    # my_sc = SublayerConnection(size)
    # # 调用对象
    # sc_result = my_sc(pe_result, sublayer)
    # print('sc_result--->', sc_result.shape, sc_result)

    # 创建前馈全连接对象 实例化前馈全连接层对象
    d_ff = 2048
    my_ff = PositionwiseFeedForward(d_model, d_ff)
    # ff_result = my_ff(mha_result)
    # # print('ff_result --->', ff_result)
    # # print('ff_result.shape --->', ff_result.shape)			# torch.Size([2, 4, 512])
    #
    # # 创建层归一化对象
    # features = d_model		# 词向量维度 512
    # eps = 1e-6				# 极小值防止分母为 0
    # my_ln = LayerNorm(features, eps)
    # ln_result = my_ln(ff_result)
    # # print('ln_result --->', ln_result)
    # # print('ln_result.shape --->', ln_result.shape)			# torch.Size([2, 4, 512])

    # # 创建编码器层对象
    # # 传入词向量维度, 多头子注意力子层对象, 前馈全连接子层, 随机失活概率
    # my_el = EncoderLayer(d_model, my_mha, my_ff, dropout_p)
    # el_result = my_el(pe_result, mask)  # 编码器最终就是输出中间语义张量C
    # print('el_result--->', el_result.shape)  # torch.Size([2, 4, 512])

    # 创建深拷贝对象，用于深拷贝编码器层中的不同的子层(多头子注意力子层、前馈前连接子层)
    #	因为要使多层编码器的每一层编码器中的每一个子层都是不同的对象，都存储在不同的内存地址中，这样就可以做到不共享参数，每一个子层都是训练自己的参数
    c = copy.deepcopy
    # 创建编码器层对象
    # 深拷贝对象, 每个对象的内存地址不同, 不共享参数
    my_el = EncoderLayer(d_model, c(my_mha), c(my_ff), dropout_p)

    # 创建编码器对象
    my_encoder = Encoder(my_el, 2)			# 创建两层编码器层
    encoder_result = my_encoder(pe_result, mask)		# 第一个编码器子层输入的就是词嵌入以及位置编码之后的结果
    # print('encoder_result--->', encoder_result.shape)
    return encoder_result		# 返回编码器输出的 中间语义张量C		(2, 4, 512)


def te03_encoder():
    vocab = 1000  # 词表大小是1000
    d_model = 512  # 词嵌入维度是512维

    # 输入x 形状是2 x 4
    x = torch.LongTensor([[100, 2, 421, 0], [491, 998, 0, 0]])

    # 输入部分的Embeddings类
    my_embeddings = Embeddings(vocab, d_model)
    embedded_result = my_embeddings(x)

    dropout_p = 0.1  # 置0概率为0.1
    max_len = 60  # 句子最大长度

    # 输入部分的PositionalEncoding类
    my_pe = PositionalEncoding(d_model, max_len, dropout_p)
    pe_result = my_pe(embedded_result)

    # 准备mask掩码张量 padding_mask
    mask = (x != 0).type(torch.uint8).unsqueeze(1).unsqueeze(2)

    head = 8
    # 创建多头注意力机制类对象
    my_mha = MultiHeadedAttention(head, d_model)

    # 创建前馈全连接对象 实例化前馈全连接层对象
    d_ff = 2048
    my_ff = PositionwiseFeedForward(d_model, d_ff)

    # 创建深拷贝对象
    c = copy.deepcopy
    # 创建编码器层对象
    # 深拷贝对象, 每个对象的内存地址不同, 不共享参数
    my_el = EncoderLayer(d_model, c(my_mha), c(my_ff), dropout_p)

    # 创建编码器对象
    my_encoder = Encoder(my_el, 2)
    encoder_result = my_encoder(pe_result, mask)
    # print('encoder_result--->', encoder_result.shape)
    return encoder_result


if __name__ == '__main__':
    encoder_result = te03_encoder()
    print(encoder_result.shape)
