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

import time

from dataclasses import dataclass

from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm


@dataclass
class config():
    batch_size = 2
    learning_rate = 2e-4
    milestones = [10, 25, 40]
    epoch_number = 50
    num_train_timesteps = 1000
    beta_schedule = "squaredcos_cap_v2"
    use_server = True
    in_channels = 2
    out_channels = 2
    image_size = 128


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


class myTransformMethod():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数
        img = cv.resize(img, (config.image_size, config.image_size))  # 改变图像大小
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将BGR(openCV默认读取为BGR)改为GRAY
        return img  # 返回预处理后的图像


if config.use_server:
    file = open('log.txt', 'w')  # 保存日志位置
else:
    file = None  # 取消日志输出

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境

train_file_list = "train_set.txt"  # 存储训练集文件名的文本文件
test_file_list = "test_set.txt"  # 存储测试集文件名的文本文件

label_path = "./BSE_JSRT"  # 标签文件夹路径
img_path = "./JSRT"  # 图像文件夹路径
myTransform = transforms.Compose(
    [myTransformMethod(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
myTrainDataset = myDataset(train_file_list, label_path, img_path, myTransform)  # 创建训练集实例
myTestDataset = myDataset(test_file_list, label_path, img_path, myTransform)  # 创建测试集实例

myTrainDataLoader = DataLoader(myTrainDataset, batch_size=config.batch_size, shuffle=True)  # 创建数据加载器实例
myTestDataLoader = DataLoader(myTestDataset, batch_size=config.batch_size, shuffle=True)  # 创建数据加载器实例

print("batch数量:", len(myTrainDataLoader))  # 输出batch数量
print("数据集大小:", len(myTrainDataset))  # 输出数据集大小

# UNet2DModel输入图像和timestep,输出图像
model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=config.in_channels,
    out_channels=config.out_channels,
    layers_per_block=3,
    block_out_channels=(64, 128, 128, 128),
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    )
)
model = model.to(device)
print("模型参数量:", sum([p.numel() for p in model.parameters()]))

# 设定噪声调度器
noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule=config.beta_schedule)
# 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
config.milestones = [x * len(myTrainDataLoader) for x in config.milestones]
# print(config.milestones)
optimizerScheduler = MultiStepLR(optimizer, milestones=config.milestones, gamma=0.1)  # 动态设置学习率

train_losses = []
test_losses = []

if __name__ == "__main__":
    # 训练过程
    print(time.strftime("%H:%M:%S", time.localtime()), "----------Begin Training----------", file=file)
    for epoch in range(config.epoch_number):
        model.train()
        print(time.strftime("%H:%M:%S", time.localtime()),
              f"Epoch:{epoch},learning rate:{optimizer.param_groups[0]['lr']}",
              file=file)
        for i, batch in tqdm(enumerate(myTrainDataLoader)):
            images, labels = batch[0].to(device), batch[1].to(device)
            blank = torch.randn_like(images).to(device)
            cat = torch.cat((labels, images), dim=-3)
            # 为图片添加噪声
            noise = torch.randn(images.shape).to(images.device)
            noise = torch.cat((noise, blank), dim=-3)
            bs = images.shape[0]

            # 为每张图片随机采样一个时间步
            timesteps = torch.randint(0, config.num_train_timesteps, (bs,), device=images.device).long()

            # 根据每个时间步的噪声幅度，向清晰的图片中添加噪声
            noisy_images = noise_scheduler.add_noise(cat, noise, timesteps)

            # 获取模型的预测结果
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # 计算损失
            train_loss = F.mse_loss(noise_pred[:,0], noise[:,0])
            train_loss.backward(train_loss)
            train_losses.append(train_loss.item())

            # 迭代模型参数
            optimizer.step()
            optimizer.zero_grad()
            # 更新动态学习率
            optimizerScheduler.step()

        if (epoch + 1) % 1 == 0:
            train_loss_last_epoch = sum(train_losses[-len(myTrainDataLoader):]) / len(myTrainDataLoader)
            print(time.strftime("%H:%M:%S", time.localtime()),
                  f"Epoch:{epoch},train losses:{train_loss_last_epoch}",
                  file=file)
        torch.save(model, "mycModel.pth")
    print(time.strftime("%H:%M:%S", time.localtime()), "----------End Training----------", file=file)
    # 查看损失曲线
    plt.plot(train_losses)  # 绘制损失曲线
    if config.use_server:
        plt.savefig("loss.png")  # 保存损失曲线
    else:
        plt.show()  # 展示损失曲线
