import torch
# m，n  和 n, k -->【m, k】
a = torch.tensor([[2, 4, 0],
                  [1, 2, 1]])
b = torch.tensor([[2, 1],
                  [0, 2],
                  [2, 4]])

# ===========二维矩阵乘法==============
print("矩阵乘法：@")
c = a @ b           # ( 2 * 3 ) @ ( 3 * 2 )
print('c',c)            # (2 * 2)       # 前一个的行为新矩阵的行，后一个的列为新矩阵的列
print("c的形状：", c.shape)
print("♥"*30)       # (2 * 2)

print("矩阵乘法：matmul()方法")
# 矩阵运算方法：torch.matmul() # 万能
d = torch.matmul(a, b)
print('d', d)
print("d的形状：", d.shape)
print("♥"*30)       # (2 * 2)

print("矩阵乘法：mm()方法： 只支持二维矩阵运算")
# 矩阵运算方法：torch.mm() # 只支持二维矩阵运算
e = torch.mm(a, b)
print('e', e)
print("e的形状：", e.shape)
print("♥"*30)


# ===========三维矩阵乘法==============
#  矩阵运算方法：torch.bmm() # 只支持三维矩阵运算
x1 = torch.randn(2, 3, 4)
print("x1：", x1)
print("♥"*30)
x2 = torch.randn(2, 4, 5)
print("x2：", x2)
print("♥"*30)

# 取第一个的 (3*4) * (4*5) = (3,5)  再取第二个的 (3*4) * (4*5) = (3,5)   最后形状为 (2,3,5)

y1 = torch.matmul(x1, x2)
print('y1', y1)
print('y1', y1.shape)       # y1 torch.Size([2, 3, 5])
print("♥"*30)

y2 = torch.bmm(x1, x2)
print('y2', y2)
print('y2', y2.shape)
print("♥"*30)               # y2 torch.Size([2, 3, 5])


# 正则表达式
import re
s = " I Love? 我 爱；you!  "
s1 = s.lower().strip()
print(s1)
s2 = re.sub(r"([?.!])", r' \1', s1)
print(s2)
s3 = re.sub(r"[^a-zA-Z.!?]+", ' ', s2)
print(s3)

# 方法返回
def fun():
    return 1, 2         # 元组形式
a = fun()
print(a)
