import torch
import torchvision

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import cv2 as cv
import pandas as pd


class myTransformMethod1():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数
        img = cv.resize(img, (28, 28))  # 改变图像大小
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将BGR(openCV默认读取为BGR)改为RGB
        return img  # 返回预处理后的图像


class myNetwork(nn.Module):  # 定义神经网络
    def __init__(self):
        super().__init__()  # 继承nn.Module的构造器
        self.flatten = nn.Flatten(-3, -1)
        # 继承nn.Module的Flatten函数并改为flatten,考虑到推理时没有batch(CHW),若使用默认值(1,-1)会导致C没有被flatten,故使用(-3,-1)
        self.linear_relu_stack = nn.Sequential(  # 定义前向传播序列
            nn.Linear(3 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):  # 定义前向传播方法
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":
    model = torch.load("./myModel.pth").to("cuda")  # 载入模型
    model.eval()  # 设置推理模式
    myTransform = transforms.Compose([myTransformMethod1(), transforms.ToTensor()])
    # 定义图像预处理组合,ToTensor()中Pytorch将HWC(openCV默认读取为height,width,channel)改为CHW,并将值[0,255]除以255进行归一化[0,1]
    for i in range(10):
        img = cv.imread("./numberImages/"+str(i)+".bmp")  # 用openCV的imread函数读取图像
        img = myTransform(img).to("cuda")  # 图像预处理
        print(torch.argmax(model(img)))
