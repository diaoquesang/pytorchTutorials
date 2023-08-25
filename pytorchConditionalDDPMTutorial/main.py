import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import cv2 as cv
import pandas as pd


class myDataset(Dataset):  # 定义数据集类
    def __init__(self, filelist, label_dir, img_dir, transform=None):  # 传入参数(标签路径,图像路径,图像预处理方式,标签预处理方式)
        self.label_dir = label_dir  # 读取标签路径
        self.img_dir = img_dir  # 读取图像路径
        self.transform = transform  # 读取图像预处理方式
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)  # 读取文件名列表

    def __len__(self):
        return len(self.filelist)  # 读取文件名数量作为数据集长度

    def __getitem__(self, idx):  # 从数据集中取出数据
        label_path = self.label_dir  # 读取标签文件夹路径
        img_path = self.img_dir  # 读取图片文件夹路径

        file = self.filelist.iloc[idx, 0]  # 读取文件名
        # print(file)
        image = cv.imread(os.path.join(img_path, file))  # 用openCV的imread函数读取图像
        label = cv.imread(os.path.join(label_path, file))  # 用openCV的imread函数读取标签

        if self.transform:
            image = self.transform(image)  # 图像预处理
            label = self.transform(label)  # 标签预处理
        return image, label  # 返回图像和标签


class resize():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数
        img = cv.resize(img, (256, 256))  # 改变图像大小
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将BGR(openCV默认读取为BGR)改为GRAY
        return img  # 返回预处理后的图像


def show_img_label_batch_tensor(img, label):  # 展示数据集函数
    img, label = img[0].numpy(), label[0].numpy()
    img, label = np.transpose(img, (1, 2, 0)), np.transpose(label, (1, 2, 0))  # C*H*W -> H*W*C
    cv.imshow("img", img)
    cv.imshow("label", label)
    cv.waitKey(0)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境
file_list = "output.txt"  # 存储文件名的文本文件
label_path = "./BSE_JSRT"  # 标签文件夹路径
img_path = "./JSRT"  # 图像文件夹路径
myTransform = transforms.Compose([resize(), transforms.ToTensor()])
myDataset = myDataset(file_list, label_path, img_path, myTransform)  # 创建数据集实例

myDataLoader = DataLoader(myDataset, batch_size=16,
                          shuffle=True)  # 创建数据加载器实例
print(len(myDataLoader))  # 输出batch数
print(len(myDataset))  # 输出数据集大小
x, y = next(iter(myDataLoader))  # 取出1个batch的图像和标签
print(x[0].shape)  # 输出图像形状
show_img_label_batch_tensor(x, y)  # 展示数据集

# UNet2DModel输入图像和timestep,输出图像
model = UNet2DModel(
    sample_size=28,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 64, 64),
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D"
    )
)
model = model.to(device)
