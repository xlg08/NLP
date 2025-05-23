# input_list = [1, 3, 2, 1, 5, 3]
#
# ngram = 2
#
# 该语句本质上是CountVectorizer对应的封装，具体的实现方法，set会对结果去重，无序
# 最后显示的词语的顺序会按照首字母顺序排序
# print(set(zip(*[input_list[i:] for i in range(ngram)])))      # 该语句本质上是CountVectorizer对应的封装


from sklearn.feature_extraction.text import CountVectorizer

# 语料
corpus = [" I love NLP NLP.",
          "NLP is fun fun fun!",
          "I study natural language processing."]

# 实例化
trans = CountVectorizer(ngram_range=(1, 1))
# 提取特征
feature = trans.fit_transform(corpus)       # 只识别两个字符之上的，才会被识别为一个词
# 词汇
print(trans.get_feature_names_out())

print(feature.toarray())
