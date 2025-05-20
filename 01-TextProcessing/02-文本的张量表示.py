'''

'''
from tensorflow.keras.preprocessing.text import Tokenizer

import joblib

# vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗", "UNK"}
# vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗", "unk"}
# vocabs = {"周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗"}
vocabs = ["周杰伦", "陈奕迅", "王力宏", "李宗盛", "吴亦凡", "鹿晗", "鹿晗"]


def dm_onehot_gen():
    # one-hot
    mytokenizer = Tokenizer()  # 词汇映射器 Tokenizer


    for vocab in vocabs:
        for text_elem in vocab:
            print(text_elem, end="\t")
            # Python 的 .lower() 方法用于将字符串中的所有大写字母转换为小写字母，
            #       返回转换后的新字符串；原字符串保持不变。
            # 这在需要忽略大小写进行比较或统一文本格式时非常有用。
            # print(text_elem.lower(), end="\t")
        print()
        # print(vocab)


    mytokenizer.fit_on_texts(texts=vocabs)

    print(mytokenizer.word_index)             # 词汇：索引  存在字典中
    print(mytokenizer.index_word)             # 索引：词汇  存在字典中
    # print(mytokenizer.index_docs)           # 字典类型，词表中每个单词出现的次数
    # print(mytokenizer.document_count)       # 返回一共有多少词汇数

    # 3 查询单词idx 赋值 zero_list，生成onehot
    for vocab in vocabs:
        # print(vocab)
        zero_list = [0] * len(vocabs)
        idx = mytokenizer.word_index[vocab] - 1
        zero_list[idx] = 1
        # print(vocab, '的onehot编码是', zero_list)

    # 4 使用joblib工具保存映射器 joblib.dump()
    mypath = './mytokenizer'
    # joblib.dump(mytokenizer, mypath)
    # print('保存mytokenizer End')


def demo_onehot():
    # one-hot
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(vocabs)
    print(tokenizer.word_index)
    print(tokenizer.index_word)

    for vocab in vocabs:
        # print(vocab)
        id = tokenizer.word_index[vocab]
        # print(id)
        zero_list = [0] * len(vocabs)
        zero_list[id - 1] = 1
        print(vocab, id, zero_list)

    joblib.dump(tokenizer, './tokenizer.pkl')


def demo_onehot_use():

    # 加载分词器
    # tokenizer = joblib.load('./tokenizer.pkl')

    tokenizer = Tokenizer(oov_token='unk')
    tokenizer.fit_on_texts(vocabs)
    print(tokenizer.word_index)
    print(tokenizer.index_word)

    # one-hot
    zero_list = [0] * len(vocabs)
    id = tokenizer.word_index['关晓彤']
    zero_list[id - 1] = 1
    print(zero_list)


if __name__ == '__main__':
    dm_onehot_gen()
    # demo_onehot()
    # demo_onehot_use()