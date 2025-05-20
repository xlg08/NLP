list1 = [1, 2, 3, 4, 5, 6]
# map: 将迭代对象的每个元素应用到自定义函数上, 返回对象(内存地址)
# 第一个参数:自定义函数名 匿名函数(func=lambda 参数:参数表达式)  x->迭代对象中每个元素,遍历
# 第二个参数:迭代对象
# obj = map(lambda x: x + 2, list1)
# print('obj-->', obj)
# list2 = list(map(lambda x: x + 2, list1))
# print('list2-->', list2)

from itertools import chain
# chain: 将多个迭代对象保存到一个迭代对象中 [1,2],[3,4],[5,6]->[1,2,3,4,5,6]
list1 = [1, 2]
list2 = [3, 4]
list3 = [[1, 2], [3, 4]]  # map函数效果
obj = map(lambda x: x, list1)
print('obj--->', obj)
# res = chain(*map(lambda x: x, list1))
# print(res)
print(*list3)
res = list(chain(*list3))
print('res--->', res)
