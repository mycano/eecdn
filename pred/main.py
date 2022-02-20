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
from tqdm import tqdm

throughput = "throughput.csv"
predict_model_path = "pred.pth"

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

def train(seq_len=5, epoch=100):
    X = []
    Y = []
    for i in range(seq_len, len(normal_rate), 1):
        X.append(normal_rate[i-seq_len:i])
        Y.append(normal_rate[i])

    # 构建测试集和训练集
    split_ratio = 0.5
    total_len = len(X)
    train_x, train_y = X[:int(split_ratio * total_len)], Y[:int(split_ratio * total_len)]
    test_x, test_y = X[int(split_ratio * total_len):], Y[int(split_ratio * total_len):]
    train_loader = DataLoader(dataset=MyDataset(train_x, train_y, transform=None))
    test_loader = DataLoader(dataset=MyDataset(test_x, test_y, transform=None), batch_size=12,
                             shuffle=False)

    criterion = nn.MSELoss()
    model = LSTM(input_size=seq_len)
    model = model.double()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.train(True)
    # train
    for i in tqdm(range(epoch), desc="seq_len: {}".format(seq_len)):
        total_loss = 0
        start_time = time.time()
        for idx, (data, label) in enumerate(train_loader):
            data1 = data.squeeze(1)
            label = label.unsqueeze(1)
            data1 = torch.divide(data1, 100)
            label = torch.divide(label, 100)
            data1 = data1.reshape((1, -1, seq_len))
            pred = model(Variable(data1))
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        end_time = time.time()
        # print("epoch: {}, loss: {:.4f}, training time: {:.4f}s".format(i+1, total_loss/len(X), end_time-start_time))

    # eval
    preds = []
    true_label = []
    model.train(False)
    torch.save(model.state_dict(), predict_model_path)
    total_loss = 0
    for idx, (x, label) in enumerate(test_loader):
        x = x.squeeze(1)  # batch_size,seq_len,input_size
        x = x.reshape(-1, 1, seq_len)
        x = torch.divide(x, 100)
        label = torch.divide(label, 100)
        pred = model(Variable(x))
        loss = criterion(pred, label.reshape(-1, 1))
        total_loss += loss
        preds.extend(pred.data.squeeze(1).tolist())
        true_label.extend(label.tolist())
    return total_loss/len(preds) * 10000


if __name__ == '__main__':
    normal_rate = np.loadtxt(throughput)
    # plt.plot(normal_rate)
    # plt.title("Recored transmission rate...")
    # plt.show()
    loss = []
    for seq_len in range(1, 11):
        _loss = train(seq_len=seq_len, epoch=100)
        loss.append(float(_loss))
    print("loss: ", loss)
    # loss =  [0.8232746276076164, 0.790592801215182, 0.826257219561161, 0.8213926874181545, 0.8350800902048686, 0.8355032428309798, 0.8378376015654219, 0.8232332321017137, 0.847506130090865, 0.8327957550734815]
    plt.bar(x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], height=loss)
    plt.ylim([min(loss)*0.95, max(loss)*1.05])
    plt.ylabel("loss")
    plt.xlabel("seq_len")
    plt.show()
    # print("eval loss is {:.4f}".format(1000*total_loss/len(preds)))
    # plt.plot(preds, label="predict")
    # plt.plot(true_label, label="true throughput")
    # plt.legend()
    # plt.show()