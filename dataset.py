import h5py
import pandas as pd
import numpy as np
file_name = "data/data.xlsx"
Train_data = pd.read_excel(file_name, sheet_name="train")
raw = Train_data.shape[0]
col = Train_data.shape[1]
train_data = np.zeros((raw, col))
for i in range(raw):
    train_data[i] = Train_data.values[i]

Test_data = pd.read_excel(file_name, sheet_name="test")
raw = Test_data.shape[0]
col = Test_data.shape[1]
test_data = np.zeros((raw, col))
for i in range(raw):
    test_data[i] = Test_data.values[i]

train_dataset = h5py.File("data/train.h5", "w")
train_dataset["train_set"] = train_data
test_dataset = h5py.File("data/test.h5", "w")
test_dataset["test_set"] = test_data


