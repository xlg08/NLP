'''

'''
import torch

list1 = [1, 2, 3]
list2 = [4, 5, 6, 7, 8, 9]
# 将多个迭代对象对应位置的元素组合在一起, 长度不一致的位置处的元素省略, 返回迭代对象
# Zip作用：两个容器对应元素进行组合（以元祖的形式），无对应元素则进行省略，并返回一个可迭代的对象
obj = zip(list1, list2)

# print(obj)
# for o in obj:
#     print(o)

obj2 = list(zip(list1, list2))
print(obj2)


# 字典反转
dic = {"a": 1, "b": 2}
my_dic = dict(zip(dic.values(), dic.keys()))
print(my_dic)

# x = torch.tensor(10, dtype=torch.long)
x = torch.tensor(10)
print(x.shape)

x = torch.tensor([10,20,30,40], dtype=torch.long)
print(x.shape)

x = torch.tensor([[10,20],[30,40]], dtype=torch.long)
print(x.shape)
