import torch
import matplotlib.pyplot as plt

# 掩码主要作用在：权重分数矩阵上，是模型暂时不去关注一些词(例如：填充的空白词、不需要关注的未出现的后面的词)
# tril():生成下三角矩阵
# triu():生成上三角矩阵
# diagonal: 移动对角线
def subsequent_mask(size):

    # 下三角
    causal_mask = torch.tril(torch.ones(size=(size, size)), diagonal=0)
    # 上三角
    # causal_mask = torch.triu(torch.ones(size=(size, size)), diagonal=0)
    return causal_mask


if __name__ == '__main__':
    causal_mask = subsequent_mask(20)
    # print('causal_mask--->', causal_mask.shape, causal_mask)

    # 绘图
    plt.figure()
    plt.imshow(causal_mask)             # 构建 20 * 20 的下三角矩阵示意图
    plt.show()

    # 模拟自回归,进行自回归掩码，前面输入的时候不能包含后面的信息
    scores = torch.randn(size=(5, 5))       # 生成 5行5列 形状的随机数
    # print("scores：", scores)
    # print("scores 的类型：", type(scores))          # <class 'torch.Tensor'>

    mask = subsequent_mask(5)       # 生成5行5列的下三角矩阵
    # print('mask==0--->', mask == 0)
    # print('mask的类型--->', type(mask))        # <class 'torch.Tensor'>
    # print('mask==0的类型--->', type(mask == 0))        # <class 'torch.Tensor'>

    # 使用 -inf 极小值进行填充
    # 将随机生成的 5行5列 形状的随机数，用下三角矩阵进行掩码处理
    masked_result = scores.masked_fill(mask == 0, value=float('-inf'))
    print('masked_result--->', masked_result)
    # print('masked_result的类型--->', type(masked_result))      # <class 'torch.Tensor'>
