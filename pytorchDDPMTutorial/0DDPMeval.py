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

class resize():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数
        img = cv.resize(img, (28, 28))  # 改变图像大小
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将BGR(openCV默认读取为BGR)改为GRAY
        img=1-img
        return img  # 返回预处理后的图像
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
noise_scheduler = DDPMScheduler(num_train_timesteps=1000,beta_schedule="squaredcos_cap_v2")
noise_scheduler.set_timesteps(1000)
myTransform = transforms.Compose([resize(), transforms.ToTensor()])
if __name__ == "__main__":
    model = torch.load("./0myDDPMModel.pth").to("cuda")  # 载入模型
    model.eval()  # 设置推理模式
    with torch.no_grad():
        CXR=cv.imread("./0train/0.png")
        CXR=myTransform(CXR).to("cuda")
        sample = torch.randn(CXR.shape).to(CXR.device)
        sample=torch.unsqueeze(sample,dim=-4)
        for j, t in enumerate(noise_scheduler.timesteps):
            print("Timesteps",j)
            residual = model(sample, t).sample
            sample = noise_scheduler.step(residual, t, sample).prev_sample
            out=torch.squeeze(sample,dim=-4)
            clean=np.array(out.detach().to("cpu"))
            clean = np.transpose(clean, (1, 2, 0))  # C*H*W -> H*W*C
            # cv.resize(clean,(100,100))
            print(clean.shape)
            cv.imshow("clean",clean)
            cv.waitKey(0)