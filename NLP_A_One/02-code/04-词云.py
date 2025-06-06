from wordcloud import WordCloud
import matplotlib.pyplot as plt
from jieba import posseg
import pandas as pd
from itertools import chain
import matplotlib

matplotlib.use('TkAgg')  # 或者 'Qt5Agg'


# 调用词云绘图
def get_word_cloud(text_list):

    # 实例化
    wordcloud = WordCloud(font_path='data/SimHei.ttf', max_words=100)
    # 拼接
    text = ' '.join(text_list)
    # 生成词云
    wordcloud.generate(text)

    # 展示
    plt.figure(figsize=(10, 10))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# 分词：形容词
def get_a_list(text):

    # 存储
    res = []
    for word in posseg.lcut(text):
        if word.flag == 'a':
            res.append(word.word)
    return res


# 处理任务
if __name__ == '__main__':

    # 读数据
    train_data = pd.read_csv('data/dev.tsv', sep='\t')

    # 获取正向的文本
    text = train_data[train_data['label'] == 0]['sentence']
    text_list = list(chain(*(map(lambda x: get_a_list(x), text))))

    get_word_cloud(text_list)
