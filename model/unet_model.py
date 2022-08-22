""" Full assembly of the parts to form the complete network """
from torch import nn

"""Refer https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py"""

import torch.nn
from model.unet_parts import Down, Up, OutConv, DoubleConv
import math
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Unet_att_ECA(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet_att_ECA, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # self.att = Res_Att_Conv_Block()
        self.c1 = ConvBlock(n_channels, 64)
        self.c1_att = ECA_Attention(n_channels)
        self.d1 = DownSample()
        self.c2 = ConvBlock(64, 128)
        self.c2_att = ECA_Attention(64)  # 每次下采样的初始通道数
        self.d2 = DownSample()
        self.c3 = ConvBlock(128, 256)
        self.c3_att = ECA_Attention(128)
        self.d3 = DownSample()
        self.c4 = ConvBlock(256, 512)
        self.c4_att = ECA_Attention(256)
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
        code_lay2 = self.c2(self.d1(code_lay1))
        code_lay2_att = self.c2_att(code_lay2)
        code_lay3 = self.c3(self.d2(code_lay2))
        code_lay3_att = self.c3_att(code_lay3)
        code_lay4 = self.c4(self.d3(code_lay3))
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

    def getName(self):
        return str(Unet_att_ECA);


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    def __init__(self):
        super(DownSample, self).__init__()
        self.layer = nn.MaxPool2d(2)

    def forward(self, x):
        return self.layer(x)


class UpSample(nn.Module):
    def __init__(self, channel):
        super(UpSample, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat((feature_map, out), dim=1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, gamma=2, b=1):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        t = int(abs((math.log(in_planes, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_y = self.avg_pool(x)
        avg_y = self.conv(avg_y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_y = self.max_pool(x)
        max_y = self.conv(max_y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        avg_out = self.sigmoid(avg_y)
        max_out = self.sigmoid(max_y)
        out = avg_out + max_out

        return out


class ChannelAttention_CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention_CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ECA_Attention(nn.Module):
    def __init__(self, in_channel):
        super(ECA_Attention, self).__init__()

        self.channel_att = ChannelAttention(in_channel)

    def forward(self, x):
        out = x
        residual = out
        out = self.channel_att(out) * out
        out += residual
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Res_Att_Conv_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Res_Att_Conv_Block, self).__init__()
        self.channel_att = ChannelAttention(in_channel)
        self.spatial_att = SpatialAttention()
        self.conv = ConvBlock(in_channel, out_channel)

    def forward(self, x):
        out = self.conv(x)
        residual = out
        out = self.channel_att(out) * out
        out = self.spatial_att(out) * out
        out += residual
        return out


class CBAM_BLOCK(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CBAM_BLOCK, self).__init__()
        self.channel_att = ChannelAttention_CBAM(in_channel)
        self.spatial_att = SpatialAttention()
        self.conv = ConvBlock(in_channel, out_channel)

    def forward(self, x):
        out = self.conv(x)
        residual = out
        out = self.channel_att(out) * out
        out = self.spatial_att(out) * out
        out += residual
        return out


class Unet_CBAM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet_CBAM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # self.att = Res_Att_Conv_Block()
        self.c1 = ConvBlock(n_channels, 64)
        self.c1_att = CBAM_BLOCK(64, 64)
        self.d1 = DownSample()
        self.c2 = ConvBlock(64, 128)
        self.c2_att = CBAM_BLOCK(128, 128)  # 每次下采样的初始通道数
        self.d2 = DownSample()
        self.c3 = ConvBlock(128, 256)
        self.c3_att = CBAM_BLOCK(256, 256)
        self.d3 = DownSample()
        self.c4 = ConvBlock(256, 512)
        self.c4_att = CBAM_BLOCK(512, 512)
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
        code_lay2 = self.c2(self.d1(code_lay1))
        code_lay2_att = self.c2_att(code_lay2)
        code_lay3 = self.c3(self.d2(code_lay2))
        code_lay3_att = self.c3_att(code_lay3)
        code_lay4 = self.c4(self.d3(code_lay3))
        code_lay4_att = self.c4_att(code_lay4)
        # 编码的最底层，也是解码的第一层
        code_lay5 = self.c5(self.d4(code_lay4))
        deCode_lay1 = self.c6(self.u1(code_lay5, code_lay4_att))
        deCode_lay2 = self.c7(self.u2(deCode_lay1, code_lay3_att))
        deCode_lay3 = self.c8(self.u3(deCode_lay2, code_lay2_att))
        deCode_lay4 = self.c9(self.u4(deCode_lay3, code_lay1_att))
        out = self.out(deCode_lay4)
        return out

    def getName(self):
        return str(Unet_CBAM);


if __name__ == '__main__':
    net = Unet_CBAM(n_channels=3, n_classes=1)
    print(net)
