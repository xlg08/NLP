from output import *


# 创建编码器解码器对象
# transformer 核心神经网络模型
# 	编码器输入层	->	编码器	-->	动态中间语义张量C
#	解码器输入层 + 编码器输出的动态中间语义张量C	-->		解码器	-->		解码器输出层	-->  根据词表维度生成输出词向量
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        # 初始化各层对象
        self.encoder = encoder  # 编码器子层对象
        self.decoder = decoder  # 解码器子层神经网络对象

        # src_embed: 编码器输入层对象 由词嵌入对象+位置编码对象组成
        # 后续代码中使用 nn.Sequential()类将词嵌入对象+位置编码对象顺序合并到一起, 顺序执行
        self.src_embed = src_embed  # 编码器输入层对象 (词嵌入对象+位置编码对象)
        # tgt_embed: 解码器输入层对象 由词嵌入对象+位置编码对象组成
        self.tgt_embed = tgt_embed  # 解码器输入层对象 (词嵌入对象+位置编码对象)

        self.generator = generator  # 解码器输出层对象

    def forward(self, src, tgt, src_mask, tgt_mask):
        # 编码器编码, 得到语义张量c
        # src为编码器输入数据（embedding和位置编码之后的结果），   src_mask 为填充掩码
        m = self.encoder(self.src_embed(src), src_mask)
        # 解码器解码
        # tgt为解码器输入数据，   src_mask 为填充掩码，    tgt_mask 为因果掩码
        x = self.decoder(self.src_embed(tgt), m, src_mask, tgt_mask)
        output = self.generator(x)  # 解码器输出层

        return output


# 创建transformer模型
# 	src_vocab：为编码器输入层词表大小
# 	tgt_vocab：为解码器输入层词表大小
# 	N：为编码器与解码器层数（或称为个数）
# 	d_model：词向量维度
# 	d_ff：前馈神经网络中第一个线性层升维后的维度
#	h：多头注意力的头数
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout_p=0.1):
    c = copy.deepcopy

    # 创建前馈全连接层
    ff = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    # 创建位置编码子层对象
    pe = PositionalEncoding(d_model=d_model, dropout_p=dropout_p)

    # 创建多头注意力子层对象
    attn = MultiHeadedAttention(h, d_model, dropout_p)

    # 创建编码器层对象
    #	解码器由 多头自注意力子层、前馈全连接子层、子层连接(残差链接+层归一化) 构成
    el = EncoderLayer(d_model, c(attn), c(ff), dropout_p)

    # 创建解码器层对象
    #	解码器由 多头自注意力子层、多头一般注意力子层、前馈全连接子层、子层连接(残差链接+层归一化) 构成
    dl = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout_p)

    # 创建模型对象
    # transformer 模型由五部分组成
    #	编码器的输入、编码器、解码器的输入、解码器、解码器的输出
    model = EncoderDecoder(
        Encoder(el, N),  # 编码器	- 由 N 个编码器子层构成
        Decoder(dl, N),  # 解码器	- 由 N 个解码器子层构成
        # 词嵌入层、位置编码器层容器
        # 创建容器列表, 后续执行按照容器中的对象顺序执行
        # nn.Sequential 是一个容器模块，按照传入的顺序执行子模块
        # 简化了前向传播的定义，无需手动编写 forward 方法
        nn.Sequential(Embeddings(src_vocab, d_model), c(pe)),  # 编码器 的输入
        nn.Sequential(Embeddings(tgt_vocab, d_model), c(pe)),  # 解码器 的输入
        Generator(d_model, tgt_vocab)  # 解码器的输出
    )

    # 将模型对象参数进行初始化
    for p in model.parameters():
        # 判断w和b的维度是否大于1维
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


if __name__ == '__main__':
    # 词表大小
    # source_vocab = 512
    # target_vocab = 512
    source_vocab = 1000  # 编码器词表大小
    target_vocab = 1000  # 解码器词表大小

    # 创建transformer模型对象
    model = make_model(source_vocab, target_vocab)
    print('model--->', model)

    # 获取模型的encoder
    print('model.encoder--->', model.encoder)
    # 获取模型encoder第1层子层
    print('model.encoder.layers[0]--->', model.encoder.layers[0])

    # 假设源数据与目标数据相同, 实际中并不相同
    source = target = torch.LongTensor([[1, 2, 3, 8], [3, 4, 1, 8]])

    # 假设src_mask(编码器掩码)与tgt_mask(解码器掩码)相同，实际中并不相同
    # 下三角矩阵
    source_mask = target_mask = torch.tril(torch.ones(size=(8, 4, 4))).type(torch.uint8)

    # 调用模型得到预测结果
    output = model(source, target, source_mask, target_mask)
    print('output--->', output.shape, output)
