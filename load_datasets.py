import sys
import numpy as np
import pandas as pd
import torch
import idx2numpy

def one_hot(label):
    # 用于把标签变成one-hot格式
    m = len(label)
    label_one_hot = torch.zeros([m,10])
    for i in range(m):
        label_one_hot[i,label[i]] = 1
    return label_one_hot

def load_data1(flag,N):
    # 用于导入csv数据，但是目前csv数据的mnist数据集在kaggle上被删掉了
    if flag=='train':
        test = pd.read_csv('mnist1/train.csv').to_numpy()
        test = torch.from_numpy(test)
        test = test[1:N+1]
        label = one_hot(test[:,0]).float()
        image = test[:,1:].float()

    if flag=='test':
        test = pd.read_csv('mnist1/test.csv').to_numpy()
        test = torch.from_numpy(test)
        test = test[1:N+1]
        label = one_hot(test[:,0]).float()
        image = test[:,1:].float()
    return image,label
    
def load_idx(images_path, labels_path):
    images = idx2numpy.convert_from_file(images_path)
    labels = idx2numpy.convert_from_file(labels_path)
    return images, labels
    
def load_data(flag,N):
    if flag=='train':
        if N>60000:
            sys.exit('Over 6e4 !')
        length = 60000
        images_path = 'mnist/train-images-idx3-ubyte'
        labels_path = 'mnist/train-labels-idx1-ubyte'
        
    if flag=='test':
        if N>10000:
            sys.exit('Over 1e4 !')
        length = 10000
        images_path = 'mnist/t10k-images-idx3-ubyte'
        labels_path = 'mnist/t10k-labels-idx1-ubyte'
        
    images, labels = load_idx(images_path, labels_path)
    images = images.reshape(length,-1)
    images_copy = np.copy(images[0:N])
    labels_copy = np.copy(labels[0:N])
    image = torch.from_numpy(images_copy).float()
    label = one_hot(torch.from_numpy(labels_copy)).float()
    return image, label

def rand_indices():
    indices = torch.randperm(20000)
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

    tr_ind, val_ind, test_ind = rand_indices()
    image_train = image_total[tr_ind].float()
    image_validation = image_total[val_ind].float()
    image_test = image_total[test_ind].float()

    if flag==1:
        label_train = pollute_label(label_total[tr_ind]).float()
    if flag==0:
        label_train = label_total[tr_ind].float()
    label_validation = label_total[val_ind].float()
    label_test = label_total[test_ind].float()
    return image_train, label_train, image_validation, label_validation, image_test, label_test
