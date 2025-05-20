"""

"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import jieba
from itertools import chain

import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'

# 数据读取：路径 分割符
train_data = pd.read_csv('data/train.tsv', sep='\t')
dev_data = pd.read_csv('data/dev.tsv', sep='\t')
# print(train_data.head())
# print(dev_data.tail())
def demo_sns_count():
    # # 标签分布统计
    # plt.figure()
    # sns.countplot(x='label',data=train_data)
    # #展示
    # plt.show()
    #
    # sns.countplot(x='label',data=dev_data)
    # plt.show()
    # train 子长度统计
    train_data['len'] = list(map(lambda x: len(x), train_data['sentence']))
    print(train_data.head())
    # sns绘图
    # sns.countplot(x='len', data=train_data)
    # plt.show()
    # sns.distplot(train_data['len'], kde=True)
    # plt.show()

    # 获取正负样本长度散点分布，也就是按照x正负样本进行分组 再按照y长度进行散点图
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))
    sns.stripplot(y='sentence_length', x='label', data=train_data)
    plt.show()

    # dev 子长度统计
    dev_data['len'] = list(map(lambda x: len(x), dev_data['sentence']))
    print(dev_data.head())
    # sns绘图
    sns.countplot(x='len', data=dev_data)
    plt.show()
    sns.distplot(dev_data['len'], kde=True)
    plt.show()

    # 获取正负样本长度散点分布，也就是按照x正负样本进行分组 再按照y长度进行散点图
    dev_data['sentence_length'] = list(map(lambda x: len(x), dev_data['sentence']))
    sns.stripplot(y='sentence_length', x='label', data=dev_data)
    plt.show()

demo_sns_count()


def dm_sns_stripplot():
    # 1 设置显示风格plt.style.use('fivethirtyeight')
    plt.style.use('fivethirtyeight')

    # 2 pd.read_csv 读训练集 验证集数据
    train_data = pd.read_csv(filepath_or_buffer='data/train.tsv', sep='\t')
    dev_data = pd.read_csv(filepath_or_buffer='data/dev.tsv', sep='\t')

    # 3 求数据长度列 然后求数据长度的分布
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))
    dev_data['sentence_length'] = list(map(lambda x: len(x), dev_data['sentence']))

    # 4 统计正负样本长度散点图 （对train_data数据，按照label进行分组，统计正样本散点图）
    sns.stripplot(y='sentence_length', x='label', data=train_data)
    plt.show()

    sns.stripplot(y='sentence_length', x='label', data=dev_data)
    plt.show()


def 数据分析():
    global train_data, dev_data
    # 数据读取：路径 分割符
    train_data = pd.read_csv('data/train.tsv', sep='\t')
    dev_data = pd.read_csv('data/dev.tsv', sep='\t')
    # print(train_data.head())
    # print(dev_data.tail())
    # # 标签分布统计
    # plt.figure()
    # sns.countplot(x='label',data=train_data)
    # #展示
    # plt.show()
    #
    # sns.countplot(x='label',data=dev_data)
    # plt.show()
    # 句子长度统计
    train_data['len'] = list(map(lambda x: len(x), train_data['sentence']))
    # print(train_data.head())
    # # sns绘图
    # sns.countplot(x='len',data=train_data)
    # plt.show()
    # sns.distplot(train_data['len'],kde=True)
    # plt.show()
    # 正负样本的散点图
    # sns.stripplot(data=train_data,x='label',y='len')
    # plt.show()
    # 统计词汇的个数
    # 分词-》map->解包-》chain-》set-》len
    # print(len(set(chain(*map(lambda x: jieba.lcut(x), train_data['sentence'])))))
    # print(len(set(chain(*map(lambda x: jieba.lcut(x), dev_data['sentence'])))))


数据分析()
