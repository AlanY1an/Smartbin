# -*- coding: utf-8 -*-
# @Time : 2024-10-2024/10/26 17:37
# @Author : 林枫
# @File : LeNet.py

import torch
from torch import nn
from torchsummary import summary
from model.config import NUM_CLASSES



class LeNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(LeNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # 使用全局平均池化来代替固定的全连接层
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出特征图大小为 1x1
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 120),  # 输入通道改为16，因为全局池化后每个通道只有一个值
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.global_avg_pool(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).cuda())

    # Check CUDA version
    cuda_version = torch.version.cuda
    print("cuDA Version:", cuda_version)

    # Check CuDNN version

    cudnn_version = torch.backends.cudnn.version()
    print("CuDNN Version:", cudnn_version)
    model = LeNet(num_classes=NUM_CLASSES).to(device)
    print(summary(model, (3, 28, 28)))
