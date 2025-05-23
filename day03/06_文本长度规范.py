'''

'''
from tensorflow.keras.preprocessing import sequence

# 数据
x_train = [[1, 23, 5, 32, 55, 63, 2, 21, 78, 32, 23, 1],
           [2, 32, 1, 23, 1]]

# value 参数 ： 用什么值进行填充，int32 类型，布尔类型也可以，浮点数会进行向下取整，选用整数部分
#   `dtype` int32 is not compatible with `value`'s type: <class 'str'>
#   You should set `dtype=object` for variable length strings.
res = sequence.pad_sequences(sequences=x_train,  maxlen=10, truncating='pre', padding='pre', value=2.2)
print(res)





