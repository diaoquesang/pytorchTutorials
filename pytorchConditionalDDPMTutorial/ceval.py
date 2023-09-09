import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DModel

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import os
import cv2 as cv
import pandas as pd

import time

from dataclasses import dataclass

from tqdm import tqdm


@dataclass
class config():
    num_train_timesteps = 1000
    beta_schedule = "squaredcos_cap_v2"
    in_channels = 2
    out_channels = 2
    image_size = 128
    see_process = True


class myTransformMethod():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数
        img = cv.resize(img, (config.image_size, config.image_size))  # 改变图像大小
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将BGR(openCV默认读取为BGR)改为GRAY
        return img  # 返回预处理后的图像


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
noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, beta_schedule=config.beta_schedule)
noise_scheduler.set_timesteps(config.num_train_timesteps)
myTransform = transforms.Compose([myTransformMethod(), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
if __name__ == "__main__":
    model = torch.load("./mycModel.pth").to("cuda")  # 载入模型
    model.eval()  # 设置推理模式
    with torch.no_grad():
        CXR = cv.imread("./JSRT/JPCLN001.png")
        CXR = cv.resize(CXR, (config.image_size, config.image_size))  # 改变图像大小

        cv.imshow("CXR", CXR)
        CXR = myTransform(CXR).to("cuda")

        sample = torch.randn(CXR.shape).to(CXR.device)
        sample = torch.cat((sample, CXR), dim=-3)
        sample = torch.unsqueeze(sample, dim=-4)
        if config.see_process:
            for j, t in tqdm(enumerate(noise_scheduler.timesteps)):
                sample = noise_scheduler.scale_model_input(sample, t)
                residual = model(sample, t).sample
                sample = noise_scheduler.step(residual, t, sample).prev_sample
                sample = torch.squeeze(sample, dim=-4)  # 2HW
                sample = sample[0]  # HW
                sample = torch.unsqueeze(sample, dim=-3)  # 1HW
                sample = torch.cat((sample, CXR), dim=-3)  # 2HW
                sample = torch.unsqueeze(sample, dim=-4)  # 12HW

                clean = torch.squeeze(sample, dim=-4)
                clean = clean[0]
                clean = torch.unsqueeze(clean, dim=-3)
                clean = np.array(clean.detach().to("cpu"))
                clean = np.transpose(clean, (1, 2, 0))  # C*H*W -> H*W*C
                clean = clean * 0.5 + 0.5
                clean = np.clip(clean, 0, 1)
                cv.imshow("clean", clean)
                cv.waitKey(0)
        else:
            for j, t in tqdm(enumerate(noise_scheduler.timesteps)):
                sample = noise_scheduler.scale_model_input(sample, t)
                residual = model(sample, t).sample
                sample = noise_scheduler.step(residual, t, sample).prev_sample
                sample = torch.squeeze(sample, dim=-4)  # 2HW
                sample = sample[0]  # HW
                sample = torch.unsqueeze(sample, dim=-3)  # 1HW
                sample = torch.cat((sample, CXR), dim=-3)  # 2HW
                sample = torch.unsqueeze(sample, dim=-4)  # 12HW
            clean = torch.squeeze(sample, dim=-4)
            clean = clean[0]
            clean = torch.unsqueeze(clean, dim=-3)
            clean = np.array(clean.detach().to("cpu"))
            clean = np.transpose(clean, (1, 2, 0))  # C*H*W -> H*W*C
            clean = clean * 0.5 + 0.5
            clean = np.clip(clean, 0, 1)
            cv.imshow("clean", clean)
            cv.waitKey(0)
        print(clean)
