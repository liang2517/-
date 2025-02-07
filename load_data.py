import h5py
import numpy as np
import torch
import torchvision.transforms


def load_dataset():
    train_dataset = h5py.File('data/train.h5', "r")
    train_data = torch.Tensor(train_dataset["train_set"])
    test_dataset = h5py.File('data/test.h5', "r")
    test_data = torch.Tensor(test_dataset["test_set"])
    return train_data, test_data

def handle_train_data(train_data):
    train_x = train_data[:, :-1]
    train_y = train_data[:, -1]
    train_y = torch.reshape(train_y,(train_data.shape[0], 1))
    return train_x, train_y
def handle_test_data(test_data):
    test_x = test_data[:, :-1]
    test_y = test_data[:, -1]
    test_y = torch.reshape(test_y,(test_data.shape[0], 1))
    return test_x, test_y
