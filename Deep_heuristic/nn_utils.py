import numpy as np
import pickle as pk
import torch
from torch.utils.data import Dataset


def load_workspace(file_reach='../training_data/reach.pk'):
    with open(file_reach, 'rb') as f:
        return pk.load(f)


def load_collision(file_reach='../training_data/direction.pk'):
    with open(file_reach, 'rb') as f:
        return pk.load(f)


def binary_acc(y_predicted, y_test):
    correct_results_sum = (torch.round(torch.sigmoid(y_predicted)) == y_test).sum().float()
    return torch.round(correct_results_sum / y_test.shape[0] * 100)


def split_data(raw_data, test_size=0.1, num_of_param=1, shuffle=False):
    raw_data = np.array(raw_data)
    if shuffle:
        np.random.shuffle(raw_data)
    data = np.array([list(x) for x in raw_data[:, 0]])
    data = np.reshape(data, (-1, num_of_param))
    train_length = int(len(data) * (1 - test_size))
    train_data_in = data[:train_length]
    train_labels_in = raw_data[:train_length, 1]
    eval_data_in = data[train_length:]
    eval_labels_in = raw_data[train_length:, 1]
    return np.array(train_data_in).reshape(-1, num_of_param), np.array(train_labels_in, dtype=float), \
        np.array(eval_data_in).reshape(-1, num_of_param), np.array(eval_labels_in, dtype=float)


class TrainData(Dataset):

    def __init__(self, x_data, y_data):
        self.X_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TestData(Dataset):

    def __init__(self, x_data):
        self.X_data = x_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)