from torch import nn
import torch
import math
from model.unet_model import ConvBlock,UpSample,DownSample


class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
    def __init__(self, c, b=1, gamma=2):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(c, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        out = x * out
        return out


class UnetECA(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(UnetECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # self.att = Res_Att_Conv_Block()
        self.c1 = ConvBlock(n_channels, 64)
        self.c1_att = EfficientChannelAttention(64)
        self.d1 = DownSample()
        self.c2 = ConvBlock(64, 128)
        self.c2_att = EfficientChannelAttention(128)  # 每次下采样的初始通道数
        self.d2 = DownSample()
        self.c3 = ConvBlock(128, 256)
        self.c3_att = EfficientChannelAttention(256)
        self.d3 = DownSample()
        self.c4 = ConvBlock(256, 512)
        self.c4_att = EfficientChannelAttention(512)
        self.d4 = DownSample()
        self.c5 = ConvBlock(512, 1024)
        self.u1 = UpSample(1024)
        self.c6 = ConvBlock(1024, 512)
        self.u2 = UpSample(512)
        self.c7 = ConvBlock(512, 256)
        self.u3 = UpSample(256)
        self.c8 = ConvBlock(256, 128)
        self.u4 = UpSample(128)
        self.c9 = ConvBlock(128, 64)
        self.out = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        code_lay1 = self.c1(x)
        code_lay1_att = self.c1_att(code_lay1)
        # code_lay1_att = x + code_lay1_att
        code_lay2 = self.c2(self.d1(code_lay1_att))
        code_lay2_att = self.c2_att(code_lay2)

        code_lay3 = self.c3(self.d2(code_lay2_att))
        code_lay3_att = self.c3_att(code_lay3)
        code_lay4 = self.c4(self.d3(code_lay3_att))
        code_lay4_att = self.c4_att(code_lay4)
        # 编码的最底层，也是解码的第一层
        code_lay5 = self.c5(self.d4(code_lay4))
        deCode_lay1 = self.c6(self.u1(code_lay5, code_lay4_att))
        deCode_lay2 = self.c7(self.u2(deCode_lay1, code_lay3_att))
        deCode_lay3 = self.c8(self.u3(deCode_lay2, code_lay2_att))
        deCode_lay4 = self.c9(self.u4(deCode_lay3, code_lay1_att))
        out = self.out(deCode_lay4)
        out = nn.Sigmoid()(out)
        return out


