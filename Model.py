from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


class CNN(nn.Module, ABC):
    def __init__(self, num_class):
        super(CNN, self).__init__()

        self.num_class = num_class

        self.conv1 = nn.Conv2d(1, 256, (3, 7), padding=(1, 3))
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 256, (3, 7), padding=(1, 3))
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 512, (3, 7), padding=(1, 3))
        self.bn3 = nn.BatchNorm2d(512)

        self.conv4 = nn.Conv2d(512, 512, (3, 7), padding=(1, 3))
        self.bn4 = nn.BatchNorm2d(512)

        self.conv5 = nn.Conv2d(512, 256, (3, 7), padding=(1, 3))
        self.bn5 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256, 26)

    def forward(self, input_shape):
        data_flow = input_shape
        data_flow = self.bn1(self.conv1(data_flow))
        data_flow = F.avg_pool2d(F.relu(data_flow), kernel_size=(1, 2))
        # data_flow = F.dropout(data_flow, p=0.25)  # 此处为dropout

        data_flow = self.bn2(self.conv2(data_flow))
        data_flow = F.avg_pool2d(F.relu(data_flow), kernel_size=(1, 2))
        # data_flow = F.dropout(data_flow, p=0.25)  # 此处为dropout

        data_flow = self.bn3(self.conv3(data_flow))
        data_flow = F.avg_pool2d(F.relu(data_flow), kernel_size=(1, 2))
        # data_flow = F.dropout(data_flow, p=0.25)  # 此处为dropout

        data_flow = self.bn4(self.conv4(data_flow))
        data_flow = F.avg_pool2d(F.relu(data_flow), kernel_size=(1, 2))
        # data_flow = F.dropout(data_flow, p=0.25)  # 此处为dropout

        data_flow = self.bn5(self.conv5(data_flow))
        data_flow = F.relu(data_flow)

        # N * C * H * W
        # F.adaptive_avg_pool2d(x, (1, 1))
        # torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        data_flow = F.adaptive_avg_pool2d(data_flow, (1, 1))
        data_flow = data_flow.view(-1, num_flat_features(data_flow))
        data_flow = self.fc(data_flow)
        # data_flow = F.softmax(data_flow, 1)

        return data_flow


class Residual(nn.Module, ABC):
    def __init__(self, num_class):
        super(Residual, self).__init__()

        self.num_class = num_class
        self.layers = nn.Sequential(
            nn.Conv2d(1, 512, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(3, 1), padding=(1, 0)),
            # nn.MaxPool2d(kernel_size=3, padding=1),
            ResidualUnit(512, 512),
            ResidualUnit(512, 512),
            ResidualUnit(512, 256),
            ResidualUnit(256, 256),
            ResidualUnit(256, 128),
            ResidualUnit(128, 128),
        )

        # self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 代替AvgPool2d以适应不同size的输入
        self.fc = nn.Linear(128, num_class)

    def forward(self, input_shape):
        data_flow = input_shape

        data_flow = self.layers(data_flow)
        data_flow = self.avg_pool(data_flow)
        data_flow = data_flow.view((input_shape.shape[0], -1))
        data_flow = self.fc(data_flow)
        return data_flow


class ResidualUnit(nn.Module, ABC):
    # 残差单元设计
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualUnit, self).__init__()
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(9, 3), stride=stride, padding=(4, 1))
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(9, 3), padding=(4, 1))

        # 　ｘ卷积后shape发生改变,比如:x:[1,64,56,56] --> [1,128,28,28],则需要1x1卷积改变x
        if in_channels != out_channels:
            self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv1x1 = None

    def forward(self, x):
        # print(x.shape)
        # o1 = self.conv1(self.relu(self.bn1(x)))
        o1 = self.relu(self.bn1(self.conv1(x)))
        # o1 = F.dropout(o1, p=0.9)  # dropout
        # print(o1.shape)
        # o2 = self.conv2(self.relu(self.bn2(o1)))
        # o2 = self.relu(self.bn2(self.conv2(o1)))
        o2 = self.bn2(self.conv2(o1))
        # o2 = F.dropout(o2, p=0.9)  # dropout
        # print(o2.shape)

        if self.conv1x1:
            x = self.conv1x1(x)

        out = self.relu(o2 + x)

        return out



# RNN Model (Many-to-One)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.25)  # batch_first=True仅仅针对输入而言
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = Variable(x.view(-1, self.sequence_length, self.input_size)).cuda()
        # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        h0 = Variable(torch.ones(self.num_layers, x.size(0), self.hidden_size).cuda())
        c0 = Variable(torch.ones(self.num_layers, x.size(0), self.hidden_size).cuda())

        # Forward propagate RNN
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
        # out = F.softmax(out, 1)
        return out