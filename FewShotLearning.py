import time

import torch
import torch.nn as nn
import learn2learn as l2l
import numpy as np
import os
from pytorchtools import EarlyStopping
from torch.autograd import Variable
from learn2learn.data.transforms import FusedNWaysKShots, RemapLabels, ConsecutiveLabels, LoadData

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# (x_train, y_train), (x_test, y_test) = rD.loadData_2d(r"C:\Users\thomas\projects\sound-pytorch\split_data")
(x_train, y_train), (x_test, y_test) = rD.loadData_2d("/home/wuyuan/sound-pytorch/split_data")
# 确认我们的电脑支持CUDA，然后显示CUDA信息：
print(device)

num_iterations = 6666
meta_batch_size = 32
batch_size = 130
split_line = 0.8
patience = 15
ways = 26
train_shots = 15
test_shots = 15
adaptation_steps = 5
fast_lr = 0.001
meta_lr = 0.1
train_mode = True
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_train.shape[2], 1))
num_classes = len(np.unique(y_train))

# 划分前两个用户的数据作为验证集
valid_index = 0
for i in range(2):
    for j in range(26):
        while y_train[valid_index] == j:
            valid_index += 1

x_valid = x_train[:valid_index, :, :, :]
y_valid = y_train[:valid_index]
x_train = x_train[valid_index:, :, :, :]
y_train = y_train[valid_index:]

# shuffle
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]

idx = np.random.permutation(len(x_valid))
x_valid = x_valid[idx]
y_valid = y_valid[idx]

idx = np.random.permutation(len(x_test))
x_test = x_test[idx]
y_test = y_test[idx]


x_train = x_train.transpose(0, 3, 1, 2)
x_valid = x_valid.transpose(0, 3, 1, 2)
x_test = x_test.transpose(0, 3, 1, 2)

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

net = CNN(26).to(device)
print(net)
maml = l2l.algorithms.MAML(net, lr=meta_lr)
if not train_mode:
    maml.load_state_dict((torch.load('maml_final.pt')))
optimizer = torch.optim.Adam(maml.parameters(), lr=fast_lr, betas=(0.9, 0.99))
criterion = nn.CrossEntropyLoss()
loss = nn.CrossEntropyLoss()

# initialize the early_stopping object
early_stopping = EarlyStopping(patience=patience, verbose=True, path="maml_best.pt")

if train_mode:

    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_transforms = [
        FusedNWaysKShots(train_dataset, n=ways, k=2 * train_shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    valid_transforms = [
        FusedNWaysKShots(valid_dataset, n=ways, k=2 * train_shots),
        LoadData(valid_dataset),
        RemapLabels(valid_dataset),
        ConsecutiveLabels(valid_dataset),
    ]
    test_dataset = l2l.data.MetaDataset(test_dataset)
    test_transforms = [
        FusedNWaysKShots(test_dataset, n=ways, k=2 * test_shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=20000)
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=5000)
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=10000)

    for iteration in range(num_iterations):

        time_start = time.time()

        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()
            batch = train_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               train_shots,
                                                               ways,
                                                               device)

            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = maml.clone()
            batch = valid_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               train_shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()



        # Average the accumulated gradients and optimize
        for p in maml.parameters():
            p.grad.data.mul_(1.0 / meta_batch_size)
        optimizer.step()

        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(meta_batch_size):
            # Compute meta-testing loss
            learner = maml.clone()
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               loss,
                                                               adaptation_steps,
                                                               test_shots,
                                                               ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

            # Print some metrics

        print('\nIteration', iteration)
        time_end = time.time()
        print('cost time %ds' % (time_end - time_start))

        print('Meta Train Error', meta_train_error / meta_batch_size)
        print('Meta Train Accuracy', meta_train_accuracy / meta_batch_size * 100)
        print('Meta Valid Error', meta_valid_error / meta_batch_size)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_batch_size * 100)
        print('Meta Test Error', meta_test_error / meta_batch_size)
        print('Meta Test Accuracy', meta_test_accuracy / meta_batch_size * 100)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(meta_valid_error / meta_batch_size, maml)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Finished Training')

    print("saving maml model ...")
    torch.save(maml.state_dict(), 'maml_final.pt')

meta_test_error = []
meta_test_accuracy = []
learner = maml.clone()
for i, data in enumerate(test_loader):

    inputs, labels = data

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
