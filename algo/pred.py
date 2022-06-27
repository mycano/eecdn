# coding: utf-8
# @Time    : 2021/1/18 15:31
# @Author  : myyao
# @FileName: pred.py
# @Software: PyCharm
# @e-mail  : myyaocn@outlook.com
# description: this file is to predict network throughput using LSTM.


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.autograd import Variable
import time
import utils


class MyDataset(Dataset):
    def __init__(self, xx, yy, transform=None):
        self.x = xx
        self.y = yy
        self.transform = transform

    def __getitem__(self, index):
        x1 = self.x[index]
        y1 = self.y[index]
        if self.transform != None:
            return self.transform(x1), y1
        return x1, y1

    def __len__(self):
        return len(self.x)


class LSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out, (hidden, cell) = self.rnn(x)
        # x.shape : batch,seq_len,hidden_size , hn.shape and cn.shape : num_layes * direction_numbers,batch,hidden_size
        a, b, c = hidden.shape
        out = self.linear(hidden.reshape(a * b, c))
        return out


if __name__ == '__main__':
    # normal_bandwidth = np.loadtxt(utils.bandwidth_csv)
    normal_bandwidth = np.loadtxt("Car/Car_1.csv")
    # plt.plot(normal_bandwidth)
    # plt.show()

    max_bandwidth = max(normal_bandwidth)
    for i in range(len(normal_bandwidth)):
        normal_bandwidth[i] = normal_bandwidth[i] / max_bandwidth

    seq_len = 5
    X = []
    Y = []
    for i in range(seq_len, len(normal_bandwidth), 1):
        X.append(normal_bandwidth[i-seq_len:i])
        Y.append(normal_bandwidth[i])
    print("bandwidth_range: [{}, {}]".format(min(Y), max(Y)))
    # 构建测试集和训练集
    split_ratio = 0.7
    total_len = len(X)
    train_x, train_y = X[:int(split_ratio * total_len)], Y[:int(split_ratio * total_len)]
    test_x, test_y = X[int(split_ratio * total_len):], Y[int(split_ratio * total_len):]
    train_loader = DataLoader(dataset=MyDataset(train_x, train_y, transform=None))
    test_loader = DataLoader(dataset=MyDataset(test_x, test_y, transform=None), batch_size=12,
                             shuffle=False)

    criterion = nn.MSELoss()
    model = LSTM()
    model = model.double()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train(True)
    # train
    for i in range(100):
        total_loss = 0
        start_time = time.time()
        for idx, (data, label) in enumerate(train_loader):
            data1 = data.squeeze(1)
            label = label.unsqueeze(1)
            data1 = data1.reshape((1, -1, seq_len))
            pred = model(Variable(data1))
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        end_time = time.time()
        print("epoch: {}, loss: {:.4f}, training time: {:.4f}s".format(i+1, total_loss/len(X), end_time-start_time))

    # eval
    preds = []
    true_label = []
    model.train(False)
    torch.save(model.state_dict(), utils.fading_predict_model_path)
    # model = model.load_state_dict(torch.load(utils.bandwidth_predict_model_path))
    total_loss = 0
    for idx, (x, label) in enumerate(test_loader):
        x = x.squeeze(1)  # batch_size,seq_len,input_size
        x = x.reshape(-1, 1, seq_len)
        # x = torch.divide(x, 100)
        # label = torch.divide(label, 100)
        pred = model(Variable(x))
        loss = criterion(pred, label.reshape(-1, 1))
        total_loss += loss
        preds.extend(pred.data.squeeze(1).tolist())
        true_label.extend(label.tolist())
    print("eval loss is {:.4f}".format(1000*total_loss/len(preds)))
    plt.plot(preds, label="predict")
    plt.plot(true_label, label="true throughput")
    plt.legend()
    plt.show()

    with open("result.txt", "w+") as f:
        f.write("true: {}\n".format(true_label))
        f.write("pred: {}\n".format(preds))

    torch.save(model, utils.fading_predict_model_path)