import torch
from Model import CNN, Residual
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import readData as rD
from pytorchtools import EarlyStopping
from sklearn.metrics import confusion_matrix
import confusionMatrix
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 确认我们的电脑支持CUDA，然后显示CUDA信息：
print(device)


epochs = 300
batch_size = 128
split_line = 0.8
patience = 15

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
(x_train, y_train), (x_test, y_test) = rD.loadData_2d(r"C:\Users\thomas\projects\sound-pytorch\split_data")
# (x_train, y_train), (x_test, y_test) = rD.loadData_2d("/home/wuyuan/sound-pytorch/split_data")
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_train.shape[2], 1))
num_classes = len(np.unique(y_train))
idx = np.random.permutation(len(x_train))

x_train = x_train[idx]
y_train = y_train[idx]

# tmp = int(split_line * x_train.shape[0])
# x_valid = x_train[tmp:, :, :, :]
# y_valid = y_train[tmp:]
# x_train = x_train[:tmp, :, :, :]
# y_train = y_train[:tmp]

x_train = x_train.transpose(0, 3, 1, 2)
# x_valid = x_valid.transpose(0, 3, 1, 2)
x_test = x_test.transpose(0, 3, 1, 2)

# 升一个维度
# y_train = y_train[:, None]

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


y_train_one_hot = to_categorical(y_train, 26)
y_test_one_hot = to_categorical(y_test, 26)

x_train_tensor = torch.from_numpy(x_train).type(torch.FloatTensor).to(device)
y_train_tensor = torch.from_numpy(y_train).type(torch.FloatTensor).to(device)

# x_valid_tensor = torch.from_numpy(x_valid).type(torch.FloatTensor).to(device)
# y_valid_tensor = torch.from_numpy(y_valid).type(torch.FloatTensor).to(device)

x_test_tensor = torch.from_numpy(x_test).type(torch.FloatTensor).to(device)
y_test_tensor = torch.from_numpy(y_test).type(torch.FloatTensor).to(device)

# TensorDataset直接包装
train_dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
# valid_dataset = torch.utils.data.TensorDataset(x_valid_tensor, y_valid_tensor)
test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

net = Residual(26).to(device)
print(net)
# 先试用交叉熵损失函数
# 在网络的forward最后输出时不用接softmax，直接全连接输出n类即可。
# 不用对标签进行one_hot编码，因为torch.nn.functional.cross_entropy里面nll_loss(negative log likelihood loss)实现的类似的过程，也就是得到对应的index。但是class = [1, 2, 3]时要处理成从0开始[0, 1, 2]
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))
# initialize the early_stopping object
early_stopping = EarlyStopping(patience=patience, verbose=True)

# 数据迭代器上循环，将数据输入给网络，并优化。
for epoch in range(epochs):
    train_losses = []
    valid_losses = []

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []


    ###################
    # train the model #
    ###################
    net.train()
    correct = 0
    total = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data

        # 梯度置0
        optimizer.zero_grad()

        # 正向传播，反向传播，优化
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        train_losses.append(loss.item())
        # # 打印状态信息
        # if i % 20 == 19:  # 每2000批次打印一次
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, np.average(train_losses)))
    print('Accuracy of the network on the train set: %d %%' % (
            100 * correct / total))

    ######################
    # validate the model #
    ######################
    net.eval()  # prep model for evaluation
    correct = 0
    total = 0
    with torch.no_grad():

        for i, data in enumerate(test_loader):
            # forward pass: compute predicted outputs by passing inputs to the model
            inputs, labels = data
            output = net(inputs)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # calculate the loss
            loss = criterion(output, labels.long())
            # record validation loss
            valid_losses.append(loss.item())



        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(epochs))

        print('Accuracy of the network on the test set: %d %%' % (
                100 * correct / total))
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)


        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        # early_stopping(valid_loss, net)
        #
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break



print('Finished Training')


correct = 0
total = 0
with torch.no_grad():
    net.eval()  # prep model for evaluation
    for i, data in enumerate(test_loader):
        inputs, labels = data

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test set: %d %%' % (
    100 * correct / total))



# labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
#           'W', 'X', 'Y', 'Z']
# # labels = ['A','B','C','D','E','F','G','H','I','J']
# # confusionMatrix.plot_confuse(model, x_test, y_test, labels)
# # model = load_model("best_model.h5")
#
# # metric = "sparse_categorical_accuracy"
# metric = "acc"
# plt.figure()
#
# plt.title("model " + metric)
# plt.ylabel(metric, fontsize="large")
# plt.xlabel("epoch", fontsize="large")
# plt.legend(["train", "val"], loc="best")
# plt.show()
#
# # # plt.close()
