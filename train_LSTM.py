# -*-coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import os
import numpy as np
import readData as rD
import torch.nn.functional as F


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# 确认我们的电脑支持CUDA，然后显示CUDA信息：
print(device)

batch_size = 256
split_line = 0.8
patience = 30

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# (x_train, y_train), (x_test, y_test) = rD.loadData_2d(r"C:\Users\thomas\projects\sound-pytorch\split_data")
(x_train, y_train), (x_test, y_test) = rD.loadData_2d("/home/wuyuan/sound-pytorch/split_data")

classes = np.unique(np.concatenate((y_train, y_test), axis=0))
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_train.shape[2], 1))

x_data = np.concatenate((x_train, x_test), axis=0)
y_data = np.concatenate((y_train, y_test), axis=0)

idx = np.random.permutation(len(x_data))
x_data = x_data[idx]
y_data = y_data[idx]

x_train = x_data[1950:]
y_train = y_data[1950:]
x_test = x_data[:1950]
y_test = y_data[:1950]

num_classes = len(np.unique(y_train))

# idx = np.random.permutation(len(x_train))
# x_train = x_train[idx]
# y_train = y_train[idx]

tmp = int(split_line * x_train.shape[0])
x_valid = x_train[tmp:, :, :, :]
y_valid = y_train[tmp:]
# x_train = x_train[:tmp, :, :, :]
# y_train = y_train[:tmp]

x_train = x_train.transpose(0, 3, 2, 1)
x_valid = x_valid.transpose(0, 3, 2, 1)
x_test = x_test.transpose(0, 3, 2, 1)

print("train set shape: ", x_train.shape)
# 升一个维度
# y_train = y_train[:, None]

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


y_train_one_hot = to_categorical(y_train, 26)
y_test_one_hot = to_categorical(y_test, 26)

x_train_tensor = torch.from_numpy(x_train).type(torch.FloatTensor).to(device)
y_train_tensor = torch.from_numpy(y_train).type(torch.FloatTensor).to(device)

x_valid_tensor = torch.from_numpy(x_valid).type(torch.FloatTensor).to(device)
y_valid_tensor = torch.from_numpy(y_valid).type(torch.FloatTensor).to(device)

x_test_tensor = torch.from_numpy(x_test).type(torch.FloatTensor).to(device)
y_test_tensor = torch.from_numpy(y_test).type(torch.FloatTensor).to(device)

# TensorDataset直接包装
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
valid_dataset = torch.utils.data.TensorDataset(x_valid_tensor, y_valid_tensor)
test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Hyper Parameters
sequence_length = 90  # 序列长度，将图像的每一列作为一个序列
input_size = 2  # 输入数据的维度
hidden_size = 64  # 隐藏层的size
num_layers = 2  # 有多少层


num_epochs = 1500
learning_rate = 0.001


# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.9)  # batch_first=True仅仅针对输入而言
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        h0 = Variable(torch.ones(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.ones(self.num_layers, x.size(0), self.hidden_size).cuda())

        # Forward propagate RNN
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
        # out = F.softmax(out, 1)
        return out

rnn = RNN(input_size, hidden_size, num_layers, num_classes)
rnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(params=rnn.parameters(), lr=0.1, momentum=0.9, dampening=0.5, weight_decay=0.01, nesterov=False)


# Train the Model
for epoch in range(num_epochs):
    train_losses = []
    valid_losses = []

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    rnn.train()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(train_loader):
        # a = images.numpy()
        images = Variable(images.view(-1, sequence_length, input_size)).cuda()  # 100*1*28*28 -> 100*28*28
        # b = images.data.cpu().numpy()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = rnn(images)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_losses.append(loss.item())

    print('Accuracy of the network on the train set: %d %%' % (
            100 * correct / total))

    rnn.eval()
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        # a = images.numpy()
        images = Variable(images.view(-1, sequence_length, input_size)).cuda()  # 100*1*28*28 -> 100*28*28
        # b = images.data.cpu().numpy()
        labels = Variable(labels).cuda()

        # Forward + Backward + Optimize
        outputs = rnn(images)
        loss = criterion(outputs, labels.long())

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        valid_losses.append(loss.item())
    print('Accuracy of the network on the test set: %d %%' % (
            100 * correct / total))
    print('[%d/%d] train loss: %f  test loss: %f' %
          (epoch + 1, num_epochs, np.average(train_losses), np.average(valid_losses)))

# Test the Model
correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data

        outputs = rnn(inputs.view(-1, sequence_length, input_size))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy of the model on the test data: %d %%' % (100 * correct / total))

# Save the Model
torch.save(rnn.state_dict(), 'rnn.pkl')