'''

'''
import re

import torch

# 为张量创建 cuda 环境，用于之后的张量都放到gpu中
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


def stringTool(oldStr: str) -> str:

    # 先将每一句话存在的大写字母都转化为小写字母，再将每一句话最前面和最后面的空白字符都去掉（例如：空格，换行，制表，回车等等，主要是换行符）
    str = oldStr.lower().strip()
    # 处理每一句话的标点符号，统一将每一句话的标点符号前面都添加一个空格
    str = re.sub(r"([.!?])", r" \1", str)
    # 将每一句话的特殊字符都转化为空格（特殊字符是除a-z的字母的以及除,!?这些标点符号的，包括空格符）,
    # 在正则表达式中要加入 + ，因为一次不只匹配一个特殊字符，多个在一起的特殊字符要只转化为一个空格
    newStr = re.sub(r"[^a-z.!?]+", r" ", str)

    # 如果最后几位是特殊字符，被转化为空格之后要将最后的空格去掉
    return newStr.strip()


if __name__ == '__main__':

    str = "I am ZhangWei ./"
    string_new = stringTool(str)
    print(string_new)
