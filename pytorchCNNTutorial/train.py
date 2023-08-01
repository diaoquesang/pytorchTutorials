import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import cv2 as cv
import pandas as pd


class myDataset(Dataset):  # 定义数据集类
    def __init__(self, annotations_file, img_dir, transform=None,
                 target_transform=None):  # 传入参数(标签路径,图像路径,图像预处理方式,标签预处理方式)
        self.img_labels = pd.read_csv(annotations_file, sep="\t", header=None)
        # 从标签路径中读取标签,sep为划分间隔符,header为列标题的行位置
        self.img_dir = img_dir  # 读取图像路径
        self.transform = transform  # 读取图像预处理方式
        self.target_transform = target_transform  # 读取标签预处理方式

    def __len__(self):
        return len(self.img_labels)  # 读取标签数量作为数据集长度

    def __getitem__(self, idx):  # 从数据集中取出数据
        img_path = self.img_labels.iloc[idx, 0]
        # 从标签对象中取出第idx行第0列(第0列为图像位置所在列)的值
        image = cv.imread(img_path)  # 用openCV的imread函数读取图像
        label = torch.tensor(eval(self.img_labels.iloc[idx, 1]))  # 从标签对象中取出第idx行第1列(第1列为图像标签所在列)的值
        if self.transform:
            image = self.transform(image)  # 图像预处理
        if self.target_transform:
            label = self.target_transform(label)  # 标签预处理
        return image, label  # 返回图像和标签


class myTransformMethod1():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数
        img = cv.resize(img, (227, 227))  # 改变图像大小
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将BGR(openCV默认读取为BGR)改为RGB
        return img  # 返回预处理后的图像


# 测试函数
# print(pd.read_csv("./d1/d1.txt", sep="\t", header=None))

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


# 设置运行环境,默认为cuda,若cuda不可用则改为mps,若mps也不可用则改为cpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")  # 输出运行环境

model = myAlexNet().to(device)  # 创建神经网络模型实例

# 设置超参数
learning_rate = 1e-4  # 学习率
batch_size = 4  # 每批数据数量
epochs = 100  # 总轮数

img_path = "./d1"  # 设置图像路径
label_path = "./d1/d1.txt"  # 设置标签路径

myTransform = transforms.Compose([myTransformMethod1(), transforms.ToTensor()])
# 定义图像预处理组合,ToTensor()中Pytorch将HWC(openCV默认读取为height,width,channel)改为CHW,并将值[0,255]除以255进行归一化[0,1]

myDataset = myDataset(label_path, img_path, myTransform)  # 创建数据集实例

myDataLoader = DataLoader(myDataset, batch_size=batch_size,
                          shuffle=True)


# 创建数据读取器(可对训练集和测试集分别创建),batch_size为每批数据数量(一般为2的n次幂以提高运行速度),shuffle为随机打乱数据

def train():
    # 记录训练过程中的损失,供后期查看
    losses = []

    # 根据epochs(总轮数)训练
    for epoch in range(epochs):
        totalLoss = 0
        # 分批读取数据
        for batch, (images, labels) in enumerate(myDataLoader):
            # 数据转换到对应运行环境
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)  # 前向传播

            # print(pred.shape)
            # print(labels.shape)

            myLoss = nn.L1Loss()  # 定义损失函数(MAE)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器

            loss = myLoss(pred, labels)  # 计算损失函数

            totalLoss += loss  # 计入总损失函数
            losses.append(loss.item())  # 存储损失,供后期查看

            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 清空梯度

            if batch % 10 == 0:  # 每隔10个batch输出1次loss
                loss, current = loss.item(), min((batch + 1) * batch_size, len(myDataset))
                print(f"epoch: {epoch:>5d} loss: {loss:>7f}  [{current:>5d}/{len(myDataset):>5d}]")

        if epoch == 0:
            minTotalLoss = totalLoss
        if totalLoss < minTotalLoss:
            print("······························模型已保存······························")
            minTotalLoss = totalLoss
            torch.save(model, "./myModel.pth")  # 保存性能最好的模型

    # 查看损失曲线
    plt.plot(losses)  # 绘制损失曲线
    plt.show()  # 展示损失曲线


if __name__ == "__main__":
    model.train()  # 设置训练模式
    train()
