import torch
from torchvision import datasets, transforms
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

#MNIST数据集：已预先划分为训练集和测试集，结构如下：训练集：60,000张图像，用于模型训练。测试集：10,000张图像，用于模型性能评估。这里调动的是原始28像素的图像
def load_MNIST(train = False, batch_size = 100):
    path = './datasets' # might need to change based on where to call this function
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  
    transform = transforms.Compose([transforms.ToTensor()])
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=False, transform=transform),
                batch_size=batch_size, shuffle=True)
        return train_loader
    else:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False, download=False, transform=transform),
                batch_size=batch_size, shuffle=False)
        return test_loader

def train_agMNIST(train = False, batch_size=100):
    if train:
        transform = transforms.Compose([transforms.ToTensor()])
        #transform = transforms.Compose([transforms.Resize((299,299)),transforms.ToTensor()])
        dataset = datasets.ImageFolder(root='./Image/ori_ag4_8_16', transform=transform)  #在这里选用要使用的训练集
        train_agloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return train_agloader

def test_semang(train=False, batch_size=100):
    if train:
        transform = transforms.Compose([transforms.ToTensor()])
        # 可添加测试专用预处理（如无需增强，保持与训练一致）
        dataset = datasets.ImageFolder(root='./datasets/semang', transform=transform)
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 测试集无需shuffle
        return test_loader
    else:
        transform = transforms.Compose([transforms.ToTensor()])
        # 可添加测试专用预处理（如无需增强，保持与训练一致）
        dataset = datasets.ImageFolder(root='./datasets/semang', transform=transform)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # 测试集无需shuffle
        return train_loader


def load_SIL(type='silhouettes'):
    '''
    Date:2021.9.9
    Function: load MNIST dataset
    Param:

    '''
    path = './datasets/SIL/'+type # might need to change based on where to call this function
    labels = os.listdir(path)
    datasets = []
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform = transforms.Compose([transforms.ToTensor()])
    for label in labels:
        for img_name in os.listdir(f"{path}/{label}"):
            img_path = f"{path}/{label}/{img_name}"
            img = Image.open(img_path)
            img = transform(img).unsqueeze(0)

            datasets.append((img, label))
    return datasets




