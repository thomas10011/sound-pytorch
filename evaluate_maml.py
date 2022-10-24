import time

import torch
import torch.nn as nn
import learn2learn as l2l
import numpy as np
import os

import readData as rD
from Model import CNN, Residual, RNN


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways, device):
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    # Separate data into adaptation/evalutation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots * ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    # Adapt the model
    for step in range(adaptation_steps):
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(adaptation_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    evaluation_error = loss(predictions, evaluation_labels)
    evaluation_accuracy = accuracy(predictions, evaluation_labels)
    return evaluation_error, evaluation_accuracy


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# (x_train, y_train), (x_test, y_test) = rD.loadData_2d(r"C:\Users\thomas\projects\sound-pytorch\split_data")
(x_train, y_train), (x_test, y_test) = rD.loadData_2d("/home/wuyuan/sound-pytorch/split_data")
# 确认我们的电脑支持CUDA，然后显示CUDA信息：
print(device)

model_name = "maml_final.pt"
num_iterations = 200
meta_batch_size = 64
batch_size = 130
split_line = 0.8
patience = 6
ways = 26
train_shots = 1
test_shots = 1
adaptation_steps = 10
fast_lr = 0.001
meta_lr = 0.1
train_mode = True
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_train.shape[2], 1))
num_classes = len(np.unique(y_train))

x_tests = []
y_tests = []
split_index = 0
for j in range(26):
    while y_test[split_index] == j:
        split_index += 1

x_test_1 = x_test[:split_index]
y_test_1 = y_test[:split_index]

x_test_2 = x_test[split_index:]
y_test_2 = y_test[split_index:]

idx = np.random.permutation(len(x_test_1))
x_test_1 = x_test_1[idx]
y_test_1 = y_test_1[idx]
idx = np.random.permutation(len(x_test_2))
x_test_2 = x_test_2[idx]
y_test_2 = y_test_2[idx]




# shuffle

idx = np.random.permutation(len(x_test))
x_test = x_test[idx]
y_test = y_test[idx]


x_test_1 = x_test_1.transpose(0, 3, 1, 2)
x_test_2 = x_test_2.transpose(0, 3, 1, 2)

x_test_1_tensor = torch.from_numpy(x_test_1).type(torch.FloatTensor)
y_test_1_tensor = torch.from_numpy(y_test_1).type(torch.FloatTensor)
x_test_2_tensor = torch.from_numpy(x_test_2).type(torch.FloatTensor)
y_test_2_tensor = torch.from_numpy(y_test_2).type(torch.FloatTensor)


# TensorDataset直接包装
test_dataset_1 = torch.utils.data.TensorDataset(x_test_1_tensor, y_test_1_tensor)
test_dataset_2 = torch.utils.data.TensorDataset(x_test_2_tensor, y_test_2_tensor)
test_datasets = [test_dataset_1, test_dataset_2]

test_loaders = []
for t in test_datasets:
    test_loaders.append(torch.utils.data.DataLoader(dataset=t, batch_size=batch_size, shuffle=True))


net = CNN(26).to(device)
print(net)
maml = l2l.algorithms.MAML(net, lr=meta_lr)
maml.load_state_dict((torch.load(model_name)))
optimizer = torch.optim.Adam(maml.parameters(), lr=fast_lr, betas=(0.9, 0.99))
criterion = nn.CrossEntropyLoss()
loss = nn.CrossEntropyLoss()


meta_test_error = []
meta_test_accuracy = []

for test_loader in test_loaders:
    learner = maml.clone()
    for i, data in enumerate(test_loader):

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # adaptation_data, adaptation_labels = inputs[:130], labels[:130]
        # evaluation_data, evaluation_labels = inputs[130:], labels[130:]

        if i == 0:
            for step in range(adaptation_steps):
                adaptation_error = loss(learner(inputs), labels.long())
                learner.adapt(adaptation_error)
        else:
            with torch.no_grad():
                predictions = learner(inputs)
                evaluation_error = loss(predictions, labels.long())
                evaluation_accuracy = accuracy(predictions, labels)

                # print('Loss of the network on the adapt batch of test set: %f' % evaluation_error)
                # print('Accuracy of the network on the adapt batch of test set: %f%%' % (100.0 * evaluation_accuracy))

                # predictions = learner(inputs)
                # evaluation_error = loss(predictions, labels.long())
                # evaluation_accuracy = accuracy(predictions, labels)

                meta_test_error.append(evaluation_error.item())
                meta_test_accuracy.append(evaluation_accuracy.item())

                print('Loss of the network on the %d batch of test set: %f' % (i, evaluation_error))
                print('Accuracy of the network on the %d batch of test set: %f%%' % (i, 100.0 * evaluation_accuracy))

    print('Average Loss of the network on the test set: %f' % (np.average(meta_test_error)))
    print('Average Accuracy of the network on the test set: %f%%' % (100.0 * np.average(meta_test_accuracy)))
