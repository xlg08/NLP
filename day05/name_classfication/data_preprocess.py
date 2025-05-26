import string
from torch.utils.data import Dataset, DataLoader
import torch

# 获取词表
all_letters = string.ascii_letters + " .,;'"
# print(all_letters)
n_letters = len(all_letters)

# 目标值
# 国家名 种类数
categories = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
              'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
# 国家名 个数
n_categories = len(categories)


# print('categories--->', categories)
# print('n_categories--->', n_categories)


# 读取数据
def read_data(file_path):
    x_list = []
    y_list = []
    # open 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        # 获取所有数据
        for line in f.readlines():
            if len(line.strip()) < 5:
                continue
            # 人名+ 国家
            x, y = line.strip().split('\t')
            # 添加数据
            x_list.append(x)
            y_list.append(y)
    return x_list, y_list


# 数据获取类
class NameClassDataset(Dataset):
    # 初始化
    def __init__(self, x_list, y_list, n_letter):
        self.x_list = x_list
        self.y_list = y_list
        self.n_letter = n_letter        # 词表大小，即：one-hot编码后词向量的维度

    # 样本数量
    def __len__(self):
        # print("__")
        return len(self.x_list)

    def __get_maxLength(self):
        max_length = 0
        for _ in self.x_list:
            if len(_) > max_length:
                max_length = len(_)
        return max_length

    # 获取某一个样本
    def __getitem__(self, idx):
        # print("2")
        '''
        :param idx: 样本索引
        :return: x,y
        '''
        idx = min(max(idx, 0), len(self.x_list) - 1)
        # 获取当前样本
        x = self.x_list[idx]
        y = self.y_list[idx]
        # 处理x,y
        tesor_y = torch.tensor(categories.index(y), dtype=torch.long)
        # x数据的处理

        max_length = self.__get_maxLength()
        # print(max_length)

        # tesor_x = torch.zeros(len(x), self.n_letter)
        # tesor_x = torch.zeros(30, self.n_letter)
        tesor_x = torch.zeros(max_length, self.n_letter)

        # print("tesor_x.shape：", tesor_x.shape)

        for index, char in enumerate(x):
            tesor_x[index][all_letters.find(char)] = 1
        return tesor_x, tesor_y


if __name__ == '__main__':

    x_list, y_list = read_data('data/name_classfication.txt')
    # print(len(x_list))
    # print(len(y_list))
    # print(categories.index('Greek'))
    # print(all_letters.find('A'))

    dataset = NameClassDataset(x_list, y_list, n_letters)

    # print(dataset[1000])

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    for x, y in dataloader:

        print(x)
        print(x.shape)
        print(y)
        print(y.shape)

        break
