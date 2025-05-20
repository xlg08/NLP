'''

'''

# 导入fasttext
import fasttext


def dm_fasttext_train_save_load():
    # 1 使用train_unsupervised(无监督训练方法) 训练词向量
    mymodel = fasttext.train_unsupervised(input='data/fil9', model='skipgram', dim=300, epoch=1)
    print('训练词向量 ok')

    # 2 save_model()保存已经训练好词向量
    # 注意，该行代码执行耗时很长
    mymodel.save_model(path="./data/fil9.bin")
    print('保存词向量 ok')

    # 3 模型加载
    mymodel = fasttext.load_model(path='./data/fil9.bin')
    print('加载词向量 ok')


# 词向量训练
def demo_w2v():
    # 训练过程
    # model = fasttext.train_unsupervised(input='data/fil9', model='skipgram', dim=200, epoch=1)
    model = fasttext.train_unsupervised(input='data/fil9', model='skipgram', dim=500, epoch=2)
    # 保存
    model.save_model(path='data/fil9.bin')


# 获取词向量
def demo_w2v_use():
    # 加载模型
    model = fasttext.load_model(path='data/fil9.bin')
    # 获取指定词的向量
    print(model.get_word_vector(word='like'))
    print(model.get_word_vector(word='like').shape)         # 查看词向量的大小
    # 获取近义词
    print(model.get_nearest_neighbors(word='like', k=5))


if __name__ == '__main__':
    # demo_w2v()
    demo_w2v_use()
