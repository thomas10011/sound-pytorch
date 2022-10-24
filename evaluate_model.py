import torch
from Model import CNN, Residual
import torch.nn as nn
import numpy as np
import readData as rD
from pytorchtools import EarlyStopping
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 确认我们的电脑支持CUDA，然后显示CUDA信息：
print(device)


epochs = 150
batch_size = 512
split_line = 0.8
patience = 150

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
(x_train, y_train), (x_test, y_test) = rD.loadData_2d(r"C:\Users\thomas\projects\sound-pytorch\split_data")
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_train.shape[2], 1))
num_classes = len(np.unique(y_train))
idx = np.random.permutation(len(x_train))

# x_train = x_train.transpose(0, 3, 1, 2)
# x_valid = x_valid.transpose(0, 3, 1, 2)
# x_test = x_test.transpose(0, 3, 1, 2)

# 升一个维度
# y_train = y_train[:, None]

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


x_test_tensor = torch.from_numpy(x_test).type(torch.FloatTensor).to(device)
y_test_tensor = torch.from_numpy(y_test).type(torch.FloatTensor).to(device)
test_dataset = torch.utils.data.TensorDataset(x_test_tensor, y_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

net = Residual(26).to(device)
net.load_state_dict((torch.load('checkpoint.pt')))
print(net)
# 先试用交叉熵损失函数
# 在网络的forward最后输出时不用接softmax，直接全连接输出n类即可。
# 不用对标签进行one_hot编码，因为torch.nn.functional.cross_entropy里面nll_loss(negative log likelihood loss)实现的类似的过程，也就是得到对应的index。但是class = [1, 2, 3]时要处理成从0开始[0, 1, 2]
criterion = nn.CrossEntropyLoss()

correct = 0
total = 0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs, labels = data

        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test set: %d %%' % (
    100 * correct / total))




