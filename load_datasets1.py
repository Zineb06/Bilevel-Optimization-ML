import sys
import numpy as np
import pandas as pd
import torch

def one_hot(label):
    # 用于把标签变成one-hot格式
    m = len(label)
    label_one_hot = torch.zeros([m,10])
    for i in range(m):
        label_one_hot[i,label[i]] = 1
    return label_one_hot

def preprocess(image):
    image_new = image/255
    return image_new

def rand_indices(N):
    indices = torch.randperm(N)
    train_size, validation_size = 5000, 5000

    train_indices = indices[:train_size]
    validation_indices = indices[train_size:(train_size + validation_size)]
    test_indices = indices[(train_size + validation_size):]
    return train_indices, validation_indices, test_indices

def pollute_label(label):
    indices = torch.randperm(5000)
    polluted_indices = indices[:2500]
    for i in polluted_indices:
        label[i] = torch.zeros(10).scatter_(0, torch.randint(10, (1,)), 1)
    return label

def load_data_from_csv(flag):
    N = 20000
    data = pd.read_csv('mnist/train.csv').to_numpy()
    data = torch.from_numpy(data)
    image_total = data[1:N+1,1:]
    label_total = one_hot(data[:,0])

    tr_ind, val_ind, test_ind = rand_indices(N)
    image_train = preprocess(image_total[tr_ind])
    image_validation = preprocess(image_total[val_ind])
    image_test = preprocess(image_total[test_ind])

    if flag==1:
        label_train = pollute_label(label_total[tr_ind]).float()
    if flag==0:
        label_train = label_total[tr_ind].float()
    label_validation = label_total[val_ind].float()
    label_test = label_total[test_ind].float()
    return (image_train, label_train), (image_validation, label_validation), (image_test, label_test)

