import numpy as np
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
        img = cv.resize(img, (227, 227))  # 改变图像大小
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将BGR(openCV默认读取为BGR)改为RGB
        return img  # 返回预处理后的图像


class myAlexNet(nn.Module):  # 定义AlexNet
    def __init__(self):
        super().__init__()  # 继承nn.Module的构造器
        self.flatten = nn.Flatten(-3, -1)
        # 卷积池化
        self.features = nn.Sequential(
            # 3x227x227(论文原输入为224x224,但考虑到padding左1右2上1下2不对称，故将输入改为227x227,放弃padding)
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            # 96x55x55
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 96x27x27
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # 256x27x27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 256x13x13
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 384x13x13
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 384x13x13
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 256x13x13
            nn.MaxPool2d(kernel_size=3, stride=2)
            # 256x6x6
        )
        # 全连接
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
            # AlexNet采取了DropOut进行正则,防止过拟合,一般取0.1,0.9,0.5三个值
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2),
            nn.Tanh()
        )

    def forward(self, x):  # 定义前向传播方法
        x = self.features(x)
        x = self.flatten(x)
        result = self.classifier(x)
        return result


if __name__ == "__main__":
    model = torch.load("./myModel.pth").to("cuda")  # 载入模型
    model.eval()  # 设置推理模式
    myTransform = transforms.Compose([myTransformMethod1(), transforms.ToTensor()])
    # 定义图像预处理组合,ToTensor()中Pytorch将HWC(openCV默认读取为height,width,channel)改为CHW,并将值[0,255]除以255进行归一化[0,1]
    for i in range(130):
        img = cv.imread("./d1/"+str(i)+".jpg")  # 用openCV的imread函数读取图像
        img = myTransform(img).to("cuda")  # 图像预处理
        print(np.array(model(img).detach().to("cpu")))
