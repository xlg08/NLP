from data_preprocess import all_letters, n_letters, n_categories, categories
import torch
from model import RNN, LSTM, GRU


# 处理要预测的输入数据
def text2oht(x):
    # 全零张量
    tensor_x = torch.zeros(len(x), n_letters)
    # 遍历命名，热编码
    for i, char in enumerate(x):
        tensor_x[i][all_letters.find(char)] = 1
    # 返回
    return tensor_x


# rnn 预测
def rnn_predict(x):
    # 热编码
    tensor_x = text2oht(x).unsqueeze(0)
    # 加载模型
    model = RNN(input_size=n_letters, hidden_size=128, output_size=n_categories)
    # 加载权重
    weight = torch.load('weight/rnn.pt')
    model.load_state_dict(weight)
    model.eval()
    # 模型预测
    with torch.no_grad():
        # 输出
        output, hn = model(tensor_x, model.initHidden(tensor_x))
        # topk
        topV, topid = output.topk(k=3, dim=1, largest=True)
        # print(topV)
        # print(topid)
        for i in range(3):
            id = topid[0][i]
            nation = categories[id]
            print(nation)


def lstm_predict(x):
    # 热编码
    tensor_x = text2oht(x).unsqueeze(0)
    # 加载模型
    model = LSTM(input_size=n_letters, hidden_size=128, output_size=n_categories)
    # 加载权重
    weight = torch.load('weight/lstm.pt')
    model.load_state_dict(weight)
    model.eval()
    # 模型预测
    with torch.no_grad():
        # 输出
        h0, c0 = model.initHidden(tensor_x)
        output, hn, cn = model(tensor_x, h0, c0)
        # topk
        topV, topid = output.topk(k=3, dim=1, largest=True)
        # print(topV)
        # print(topid)
        for i in range(3):
            id = topid[0][i]
            nation = categories[id]
            print(nation)


# gru 预测
def gru_predict(x):
    # 热编码
    tensor_x = text2oht(x).unsqueeze(0)
    # 加载模型
    model = GRU(input_size=n_letters, hidden_size=128, output_size=n_categories)
    # 加载权重
    weight = torch.load('weight/gru.pt')
    model.load_state_dict(weight)
    model.eval()
    # 模型预测
    with torch.no_grad():
        # 输出
        output, hn = model(tensor_x, model.initHidden(tensor_x))
        # topk
        topV, topid = output.topk(k=3, dim=1, largest=True)
        # print(topV)
        # print(topid)
        for i in range(3):
            id = topid[0][i]
            nation = categories[id]
            print(nation)


def dm_test_predic_rnn_lstm_gru():
    # 把三个函数的入口地址 组成列表，统一输入数据进行测试
    for func in [rnn_predict, lstm_predict, gru_predict]:
        func('XvLong')


if __name__ == '__main__':
    x = 'Zhangsan'
    rnn_predict(x)
    # lstm_predict(x)
    # gru_predict(x)
    # dm_test_predic_rnn_lstm_gru()
