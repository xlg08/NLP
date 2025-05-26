import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data_preprocess import *
from model import *
from torch.optim import Adam
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt



# 流程
"""
1.超参数
2.数据
3.模型
4.优化器
5.损失
6.遍历每个轮次
7.遍历每个批次
8.模型预测
9.计算损失
10.梯度清零
11.反向传播
12.更新参数
13.模型保存
"""


# RNN的过程
def train_rnn():

    # 1.超参数
    lr = 0.001
    epochs = 10
    batch_size = 5

    # 2.数据
    x_list, y_list = read_data('data/name_classfication.txt')
    dataset = NameClassDataset(x_list, y_list, n_letters)

    # 3.模型
    # rnn = RNN(input_size=n_letters, hidden_size=64, output_size=n_categories)
    rnn = RNN(input_size=n_letters, hidden_size=128, output_size=n_categories)

    # 4.优化器
    optim = Adam(rnn.parameters(), lr=lr)

    # 5.损失
    ce_loss = NLLLoss()
    starttime = time.time()
    total_iter_num = 0  # 已训练的样本数
    total_loss = 0.0  # 已训练的损失和
    total_loss_list = []  # 每100个样本求一次平均损失 形成损失列表
    total_acc_num = 0  # 已训练样本预测准确总数
    total_acc_list = []  # 每100个样本求一次平均准确率 形成平均准确率列表

    # 6.遍历每个轮次
    for epoch in range(epochs):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # 7.遍历每个批次
        for i, (x_batch, y_batch) in enumerate(dataloader):

            # 8.模型预测
            h0 = rnn.initHidden(x_batch)
            # output, hn = rnn(x_batch[0], h0)
            output, hn = rnn(x_batch, h0)

            # 9.计算损失
            loss = ce_loss(output, y_batch)

            # 反向传播，更新参数
            # 10.梯度清零
            optim.zero_grad()
            # 11.反向传播
            loss.backward()
            # 12.更新参数
            optim.step()

            # 损失
            # total_iter_num = total_iter_num + 1
            total_iter_num = total_iter_num + x_batch.shape[0]

            # total_loss = total_loss + loss.item()
            total_loss += loss.item()

            # 统计正确的个数

            for _ in range(x_batch.shape[0]):
                i_predict_tag = (1 if torch.argmax(output[_], dim=-1).item() == y_batch[_].item() else 0)
                total_acc_num = total_acc_num + i_predict_tag

            # i_predict_tag = (1 if torch.argmax(output, dim=-1).item() == y_batch.item() else 0)
            # total_acc_num = total_acc_num + i_predict_tag

            if (total_iter_num % 10 == 0):
                # 平均损失
                tmploss = total_loss / total_iter_num
                total_loss_list.append(tmploss)
                # 平均准确率
                tmpacc = total_acc_num / total_iter_num
                total_acc_list.append(tmpacc)

            if (total_iter_num % 20 == 0):
                tmploss = total_loss / total_iter_num
                tmpacc = total_acc_num / total_iter_num
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' % (
                    epoch + 1, tmploss, time.time() - starttime, tmpacc))
            # todo:6-3每个轮次保存模型

    # 13.模型保存
    torch.save(rnn.state_dict(), 'weight/rnn.pt')
    total_time = time.time() - starttime
    return total_loss_list, total_acc_list, total_time

def train_lstm():

    # 1.超参数
    lr = 0.001
    epochs = 10
    batch_size = 5

    # 2.数据
    x_list, y_list = read_data('data/name_classfication.txt')
    dataset = NameClassDataset(x_list, y_list, n_letters)

    # 3.模型
    lstm = LSTM(input_size=n_letters, hidden_size=128, output_size=n_categories)

    # 4.优化器
    optim = Adam(lstm.parameters(), lr=lr)

    # 5.损失
    ce_loss = NLLLoss()
    starttime = time.time()
    total_iter_num = 0  # 已训练的样本数
    total_loss = 0.0  # 已训练的损失和
    total_loss_list = []  # 每100个样本求一次平均损失 形成损失列表
    total_acc_num = 0  # 已训练样本预测准确总数
    total_acc_list = []  # 每100个样本求一次平均准确率 形成平均准确率列表

    # 6.遍历每个轮次
    for epoch in range(epochs):

        lstm.train()

        # 7.遍历每个批次
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i, (x_batch, y_batch) in enumerate(dataloader):
            # 8.模型预测
            h0, c0 = lstm.initHidden(x_batch)
            # output, hn, cn = lstm(x_batch[0], h0, c0)
            output, hn, cn = lstm(x_batch, h0, c0)
            # 9.计算损失
            loss = ce_loss(output, y_batch)

            # 10.梯度清零
            optim.zero_grad()
            # 11.反向传播
            loss.backward()
            # 12.更新参数
            optim.step()

            # 损失
            # total_iter_num = total_iter_num + 1
            total_iter_num += x_batch.shape[0]
            total_loss = total_loss + loss.item()

            # 统计正确的个数
            for _ in range(x_batch.shape[0]):
                i_predict_tag = (1 if torch.argmax(output[_], dim=-1).item() == y_batch[_].item() else 0)
                total_acc_num = total_acc_num + i_predict_tag

            if (total_iter_num % 10 == 0):
                # 平均损失
                tmploss = total_loss / total_iter_num
                total_loss_list.append(tmploss)
                # 平均准确率
                tmpacc = total_acc_num / total_iter_num
                total_acc_list.append(tmpacc)
            if (total_iter_num % 20 == 0):
                tmploss = total_loss / total_iter_num
                tmpacc = total_acc_num / total_iter_num
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' % (
                    epoch + 1, tmploss, time.time() - starttime, tmpacc/batch_size))

    # 13.模型保存
    torch.save(lstm.state_dict(), 'weight/lstm.pt')
    total_time = time.time() - starttime
    return total_loss_list, total_acc_list, total_time

def train_gru():

    # 1.超参数
    lr = 0.001
    epochs = 10
    batch_size = 5

    # 2.数据
    x_list, y_list = read_data('data/name_classfication.txt')
    dataset = NameClassDataset(x_list, y_list, n_letters)

    # 3.模型
    gru = GRU(input_size=n_letters, hidden_size=128, output_size=n_categories)

    # 4.优化器
    optim = Adam(gru.parameters(), lr=lr)

    # 5.损失
    ce_loss = NLLLoss()
    starttime = time.time()
    total_iter_num = 0  # 已训练的样本数
    total_loss = 0.0  # 已训练的损失和
    total_loss_list = []  # 每100个样本求一次平均损失 形成损失列表
    total_acc_num = 0  # 已训练样本预测准确总数
    total_acc_list = []  # 每100个样本求一次平均准确率 形成平均准确率列表

    # 6.遍历每个轮次
    for epoch in range(epochs):

        # 7.遍历每个批次
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for i, (x_batch, y_batch) in enumerate(dataloader):
            # 8.模型预测
            h0 = gru.initHidden(x_batch)
            # output, hn = gru(x_batch[0], h0)
            output, hn = gru(x_batch, h0)
            # 9.计算损失
            loss = ce_loss(output, y_batch)

            # 10.梯度清零
            optim.zero_grad()
            # 11.反向传播
            loss.backward()
            # 12.更新参数
            optim.step()

            # 损失
            total_iter_num = total_iter_num + x_batch.shape[0]
            total_loss = total_loss + loss.item()

            # 统计正确的个数
            # i_predict_tag = (1 if torch.argmax(output, dim=-1).item() == y_batch.item() else 0)
            # total_acc_num = total_acc_num + i_predict_tag
            # 统计正确的个数
            for _ in range(x_batch.shape[0]):
                i_predict_tag = (1 if torch.argmax(output[_], dim=-1).item() == y_batch[_].item() else 0)
                total_acc_num = total_acc_num + i_predict_tag

            if (total_iter_num % 10 == 0):
                # 平均损失
                tmploss = total_loss / total_iter_num
                total_loss_list.append(tmploss)
                # 平均准确率
                tmpacc = total_acc_num / total_iter_num
                total_acc_list.append(tmpacc)
            if (total_iter_num % 20 == 0):
                tmploss = total_loss / total_iter_num
                tmpacc = total_acc_num / total_iter_num
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' % (
                    epoch + 1, tmploss, time.time() - starttime, tmpacc))
    # 13.模型保存
    torch.save(gru.state_dict(), 'weight/gru.pt')
    total_time = time.time() - starttime
    return total_loss_list, total_acc_list, total_time


# 三个网络一起进行训练
def train():

    # 返回对应信息：损失、准确率、时间
    rnn_loss, rnn_acc, rnn_time = train_rnn()
    lstm_loss, lstm_acc, lstm_time = train_lstm()
    gru_loss, gru_acc, gru_time = train_gru()

    # 可视化
    # 损失折线图
    plt.figure(0)
    plt.plot(rnn_loss,label='RNN')
    plt.plot(lstm_loss,label='lstm')
    plt.plot(gru_loss,label='gru')
    plt.legend()
    plt.savefig('img/loss.png')
    plt.show()
    # 准确率折线图
    plt.figure(1)
    plt.plot(rnn_acc, label='RNN')
    plt.plot(lstm_acc, label='lstm')
    plt.plot(gru_acc, label='gru')
    plt.legend()
    plt.savefig('img/acc.png')
    plt.show()
    # 时间
    plt.figure(2)
    plt.bar(range(3), [rnn_time,lstm_time,gru_time],tick_label=['rnn','lstm','gru_time'])
    plt.savefig('img/time.png')
    plt.show()


if __name__ == '__main__':

    # train_rnn()
    # train_lstm()
    # train_gru()

    train()