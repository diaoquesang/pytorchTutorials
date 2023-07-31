import torch
import torchvision

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
        self.img_labels = pd.read_csv(annotations_file, sep=" ", header=None)
        # 从标签路径中读取标签,sep为划分间隔符,header为列标题的行位置
        self.img_dir = img_dir  # 读取图像路径
        self.transform = transform  # 读取图像预处理方式
        self.target_transform = target_transform  # 读取标签预处理方式

    def __len__(self):
        return len(self.img_labels)  # 读取标签数量作为数据集长度

    def __getitem__(self, idx):  # 从数据集中取出数据
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[
            idx, 0])
        # 从标签对象中取出第idx行第0列(第0列为图像位置所在列)的值(numberImages\5.bmp),并与图像路径(numberImages)进行拼接
        image = cv.imread(img_path)  # 用openCV的imread函数读取图像
        label = self.img_labels.iloc[idx, 1]  # 从标签对象中取出第idx行第1列(第1列为图像标签所在列)的值(5)
        if self.transform:
            image = self.transform(image)  # 图像预处理
        if self.target_transform:
            label = self.target_transform(label)  # 标签预处理
        return image, label  # 返回图像和标签


class myTransformMethod1():  # Python3默认继承object类
    def __call__(self, img):  # __call___，让类实例变成一个可以被调用的对象，像函数
        img = cv.resize(img, (28, 28))  # 改变图像大小
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将BGR(openCV默认读取为BGR)改为RGB
        return img  # 返回预处理后的图像

# 测试函数
# print(pd.read_csv("annotations.txt", sep=" ", header=None))
# print(os.path.join("numberImages", pd.read_csv("annotations.txt", sep=" ", header=None).iloc[5, 0]))
# print(pd.read_csv("annotations.txt", sep=" ", header=None).iloc[5, 1])
# cv.imshow("1",cv.imread(os.path.join("numberImages", pd.read_csv("annotations.txt", sep=" ", header=None).iloc[5, 0])))
# cv.waitKey(0)


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


# 设置运行环境,默认为cuda,若cuda不可用则改为mps,若mps也不可用则改为cpu
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")  # 输出运行环境

model = myNetwork().to(device)  # 创建神经网络模型实例

# 设置超参数
learning_rate = 1e-5  # 学习率
batch_size = 8  # 每批数据数量
epochs = 3000  # 总轮数

img_path = "./numberImages"  # 设置图像路径
label_path = "./annotations.txt"  # 设置标签路径

myTransform = transforms.Compose([myTransformMethod1(), transforms.ToTensor()])
# 定义图像预处理组合,ToTensor()中Pytorch将HWC(openCV默认读取为height,width,channel)改为CHW,并将值[0,255]除以255进行归一化[0,1]

myDataset = myDataset(label_path, img_path, myTransform)  # 创建数据集实例

myDataLoader = DataLoader(myDataset, batch_size=batch_size,
                          shuffle=True)
# 创建数据读取器(可对训练集和测试集分别创建),batch_size为每批数据数量(一般为2的n次幂以提高运行速度),shuffle为随机打乱数据

def train():
    # 根据epochs(总轮数)训练
    for epoch in range(epochs):
        totalLoss = 0
        # 分批读取数据
        for batch, (images, labels) in enumerate(myDataLoader):
            # 数据转换到对应运行环境
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)  # 前向传播

            myLoss = nn.CrossEntropyLoss()  # 定义损失函数(交叉熵)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器

            loss = myLoss(pred, labels)  # 计算损失函数

            totalLoss += loss  # 计入总损失函数

            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 清空梯度

            if batch % 1 == 0:  # 每隔1个batch输出1次loss
                loss, current = loss.item(), min((batch + 1) * batch_size,len(myDataset))
                print(f"epoch: {epoch:>5d} loss: {loss:>7f}  [{current:>5d}/{len(myDataset):>5d}]")

        if epoch == 0:
            minTotalLoss = totalLoss
        if totalLoss < minTotalLoss:
            print("······························模型已保存······························")
            minTotalLoss = totalLoss
            torch.save(model, "./myModel.pth")  # 保存性能最好的模型


if __name__ == "__main__":
    model.train()  # 设置训练模式
    train()
