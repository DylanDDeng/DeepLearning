# -*- coding = utf-8 -*-  
# @Time: 2023/3/3 15:18 
# @Author: Dylan 
# @File: utils.py 
# @software: PyCharm
import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt

def read_split_data(root: str, val_rate: float = 0.2, plot_image: bool = False):
    """
    split data set
    :param plot_image: bool value to decide if plot the image
    :param root: the data path
    :param val_rate: the validation set rate. Default value is 0.2
    :return: train_images_path, train_images_label, val_images_path, val_images_label
    """
    random.seed(0)   # set random seed

    assert os.path.exists(root), "dataset root:{} does not exist".format(root)

    # 遍历文件夹
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序
    flower_class.sort()

    # 生成类别名称以及对应的索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # path to restore train data set
    train_images_label = []  # restore train dataset label
    val_images_path = []  # restore val dataset  path
    val_images_label = []  # restore val dataset label
    every_class_num = []  # the number of data in every class
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # support document

    # 遍历文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)

        # 遍历获取supported 支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]  # every image paths
        # 排序
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # record the number of data in every class
        every_class_num.append(len(images))
        # 按比例随机采集样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)

            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{}images were found in the dataset.".format(sum(every_class_num)))
    print('{}images for training.'.format(len(train_images_path)))
    print('{}images for validation.'.format(len(val_images_label)))

    assert len(train_images_path) > 0, "number of training must greater than 0"
    assert len(val_images_path) > 0, "number of validation set must greater than 0"

    # plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # x column name
        plt.xticks(range(len(flower_class)), flower_class)

        # add value on the bar
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v+5, s=str(v), ha='center')

        # set x axis label
        plt.xlabel('image_class')
        # set y axis label
        plt.ylabel('number of images')
        # title
        plt.title('flower class distribution')

        plt.show()
        plt.savefig('./flower_class.png')

    return train_images_path, train_images_label, val_images_path, val_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + 'does not exist'
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            labels = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(labels)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))

        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
        data_loader.desc = "[epoch{}] mean loss{}".format(epoch, round(mean_loss.item(), 3))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, end training', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return mean_loss.item()

@ torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    # check the dataset num
    total_num = len(data_loader.dataset)

    # num of predict right
    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    return sum_num.item()/total_num




















