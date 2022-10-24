import numpy as np


def readFilePath():
    file = open("D:\\学习\\研三\论文\\filepath.txt", "r")
    paths = []
    while 1:
        line = file.readline()
        if not line:
            break
        line = line.replace('\n', '').replace('\r', '')
        paths.append(line)
    file.close()
    return paths


def readData(path):
    file = open(path, "r")
    data = []
    while 1:
        line = file.readline()
        if not line:
            break
        line = line[:-2]
        line = line.replace('\n', '').replace('\r', '')
        list = line.split(',')
        raw = []
        for item in list:
            raw.append(float(item))
        data.append(raw)
    result = np.array(data)
    print("result shape:", result.shape)
    file.close()
    return result


def readData_2d(path):
    file = open(path, "r")
    data = []
    while 1:
        line = file.readline()
        if not line:
            break
        line = line[:-2]
        line = line.replace('\n', '').replace('\r', '')
        list = line.split(',')
        raw = []
        for item in list:
            raw.append(float(item))
        newRaw = [[], []]
        for i in range(int(len(raw) / 2)):
            newRaw[0].append(raw[i])
        for i in range(int(len(raw) / 2), len(raw)):
            newRaw[1].append(raw[i])
        data.append(newRaw)
    result = np.array(data)
    print("result shape:", result.shape)
    file.close()
    return result


def getLabel(path):
    file = open(path, "r")
    data = []
    while 1:
        line = file.readline()
        if not line:
            break
        line = line.replace('\n', '').replace('\r', '')
        data.append(int(line))
    return np.array(data)


def data_fill(X):
    X = X.tolist()
    # max_len = int(max([len(arr) for arr in X]) / 2)
    max_len = 225
    X_1 = []
    X_2 = []
    for arr in X:
        arr_len = len(arr)
        X_1.append(arr[0:int(arr_len / 2)])
        X_2.append(arr[int(arr_len / 2):arr_len])
    X_1 = np.array(X_1)
    X_2 = np.array(X_2)
    X_1_padded = np.array(
        [np.lib.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=0) for arr in X_1])
    X_2_padded = np.array(
        [np.lib.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=0) for arr in X_2])
    result = np.hstack((X_1_padded, X_2_padded))
    return result


def loadData_2d(dataPath):
    path = dataPath + "\\train_data"
    X_train = readData_2d(path)
    path = dataPath + "\\train_label"
    y_train = getLabel(path)
    path = dataPath + "\\test_data"
    X_test = readData_2d(path)
    path = dataPath + "\\test_label"
    y_test = getLabel(path)
    print(X_train.shape, len(y_train), len(X_test[0]), len(y_test))
    # X_train_padded = data_fill(X_train)
    # X_test_padded = data_fill(X_test)
    # return (X_train_padded, y_train), (X_test_padded, y_test)
    return (X_train, y_train), (X_test, y_test)
    # return (X_train[:,80:160],y_train),(X_test[:,80:160],y_test)


def loadData(dataPath):
    path = dataPath + "\\train_data"
    X_train = readData(path)
    path = dataPath + "\\train_label"
    y_train = getLabel(path)
    path = dataPath + "\\test_data"
    X_test = readData(path)
    path = dataPath + "\\test_label"
    y_test = getLabel(path)
    print(X_train.shape, len(y_train), len(X_test[0]), len(y_test))
    # X_train_padded = data_fill(X_train)
    # X_test_padded = data_fill(X_test)
    # return (X_train_padded, y_train), (X_test_padded, y_test)
    return (X_train, y_train), (X_test, y_test)
    # return (X_train[:,80:160],y_train),(X_test[:,80:160],y_test)

# loadData("D:\\学习\\研三\\论文\\data")
# (data_input, data_label), (x_test, y_test) = loadData("D:\\学习\\研三\\论文\\data\\train_data_double")
