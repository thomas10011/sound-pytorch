import random
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


def split_user_data(data, label, shuffle=True):
    index = 0
    start_index = 0
    split_dataset = []
    while index < len(label):
        for j in range(26):
            while label[index] == j:
                index += 1
                if len(label) == index:
                    break
        end_index = index
        tmp_x = data[start_index:end_index]
        tmp_y = label[start_index:end_index]

        tmp_x = tmp_x.transpose(0, 3, 1, 2)

        if shuffle:
            tmp_x, tmp_y = shuffle_data(tmp_x, tmp_y)
        split_dataset.append((tmp_x, tmp_y))
        start_index = end_index
    return split_dataset


def shuffle_data(data, label):
    index = np.random.permutation(len(label))
    return data[index], label[index]


def convert2tensor_dataset(data, label):
    data_tensor = torch.from_numpy(data).type(torch.FloatTensor).to(device)
    label_tensor = torch.from_numpy(label).type(torch.FloatTensor).to(device)
    dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)
    return dataset


def transform2meta_dataset(dataset, ways, shots, num_tasks):
    dataset = l2l.data.MetaDataset(dataset)
    transforms = [
        FusedNWaysKShots(dataset, n=ways, k=2 * shots),
        LoadData(dataset),
        RemapLabels(dataset),
        ConsecutiveLabels(dataset),
    ]
    tasks = l2l.data.TaskDataset(dataset,
                                 task_transforms=transforms,
                                 num_tasks=num_tasks)
    return tasks


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# (x_train, y_train), (x_test, y_test) = rD.loadData_2d(r"C:\Users\thomas\projects\sound-pytorch\split_data")
(x_train, y_train), (x_test, y_test) = rD.loadData_2d("/home/wuyuan/sound-pytorch/split_data")
# 确认我们的电脑支持CUDA，然后显示CUDA信息：
print(device)

num_iterations = 6666
meta_batch_size = 128
batch_size = 130
split_line = 0.8
patience = 50
ways = 26
train_shots = 1
valid_shots = 1
test_shots = 1
adaptation_steps = 1
fast_lr = 0.001
meta_lr = 0.1
train_mode = True
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_train.shape[2], 1))
num_classes = len(np.unique(y_train))

train_dataset = split_user_data(x_train, y_train)
test_dataset = split_user_data(x_test, y_test)


# TensorDataset直接包装
train_loaders, valid_loaders, test_loaders = [], [], []
train_datasets, valid_datasets, test_datasets = [], [], []
for i, (data, label) in enumerate(train_dataset):
    dataset = convert2tensor_dataset(data, label)
    train_datasets.append(dataset)
    train_loaders.append(torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True))

for i, (data, label) in enumerate(test_dataset):
    dataset = convert2tensor_dataset(data, label)
    test_datasets.append(dataset)
    test_loaders.append(torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True))

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
    train_meta_datasets, valid_meta_datasets, test_meta_datasets = [], [], []
    for dataset in train_datasets:
        task = transform2meta_dataset(dataset, ways=ways, shots=train_shots, num_tasks=10000)
        train_meta_datasets.append(task)

    for dataset in test_datasets:
        task = transform2meta_dataset(dataset, ways=ways, shots=test_shots, num_tasks=7000)
        test_meta_datasets.append(task)


    for iteration in range(num_iterations):
        time_start = time.time()

        random.shuffle(train_meta_datasets)

        optimizer.zero_grad()
        meta_train_error = []
        meta_train_accuracy = []
        meta_valid_error = []
        meta_valid_accuracy = []

        train_len = len(train_meta_datasets)
        for i in range(1, train_len):

            valid_meta_dataset = train_meta_datasets[i]

            for j in range(meta_batch_size):
                train_index = j % train_len
                # 跳过验证集
                if train_index == i:
                    continue
                # Compute meta-training loss
                learner = maml.clone()
                # 每一次迭代只在一个用户的数据上进行迭代更新
                train_tasks = train_meta_datasets[train_index]

                batch = train_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   loss,
                                                                   adaptation_steps,
                                                                   train_shots,
                                                                   ways,
                                                                   device)

                evaluation_error.backward()
                meta_train_error.append(evaluation_error.item())
                meta_train_accuracy.append(evaluation_accuracy.item())

                valid_tasks = valid_meta_dataset
                # Compute meta-validation loss
                learner = maml.clone()
                batch = valid_tasks.sample()
                evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                                   learner,
                                                                   loss,
                                                                   adaptation_steps,
                                                                   valid_shots,
                                                                   ways,
                                                                   device)
                meta_valid_error.append(evaluation_error.item())
                meta_valid_accuracy.append(evaluation_accuracy.item())

            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            optimizer.step()

        meta_test_error = []
        meta_test_accuracy = []
        for test_tasks in test_meta_datasets:
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
                meta_test_error.append(evaluation_error.item())
                meta_test_accuracy.append(evaluation_accuracy.item())

            # Print some metrics

        print('\nIteration', iteration)
        time_end = time.time()
        print('cost time %ds' % (time_end - time_start))

        print('Meta Train Error', np.average(meta_train_error))
        print('Meta Train Accuracy', np.average(meta_train_accuracy) * 100)
        print('Meta Valid Error', np.average(meta_valid_error))
        print('Meta Valid Accuracy', np.average(meta_valid_accuracy) * 100)
        print('Meta Test Error', np.average(meta_test_error))
        print('Meta Test Accuracy', np.average(meta_test_accuracy) * 100)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(np.average(meta_valid_error), maml)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Finished Training')

    print("saving maml model ...")
    torch.save(maml.state_dict(), 'maml_final.pt')

meta_test_error = []
meta_test_accuracy = []
learner = maml.clone()
for test_loader in test_loaders:
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
