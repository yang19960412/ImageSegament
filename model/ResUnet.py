from torch import nn
import torch
from model.unet_model import DownSample, ConvBlock, UpSample

class ConvBlock1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock1, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.layer(x)

class ResDoubleConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResDoubleConvBlock, self).__init__()
        self.conv = ConvBlock1(in_channel, out_channel)
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=1,padding=0)
        self.relu = nn.LeakyReLU()
    def forward(self, x):
        residual = self.conv1(x)
        out = self.conv(x)
        out = residual + out
        out = self.relu(out)
        return out


class ResUnet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(ResUnet, self).__init__()
        self.code_conv1 = ConvBlock(in_channel=in_channel, out_channel=64)
        self.code_down1 = DownSample()
        self.code_conv2 = ResDoubleConvBlock(in_channel=64, out_channel=128)
        self.code_down2 = DownSample()
        self.code_conv3 = ResDoubleConvBlock(in_channel=128, out_channel=256)
        self.code_down3 = DownSample()
        self.code_conv4 = ResDoubleConvBlock(in_channel=256, out_channel=512)
        self.code_down4 = DownSample()
        self.code_conv5 = ResDoubleConvBlock(in_channel=512, out_channel=1024)
        self.de_up1 = UpSample(in_channel=1024)
        self.de_conv1 = ResDoubleConvBlock(in_channel=1024, out_channel=512)
        self.de_up2 = UpSample(in_channel=512)
        self.de_conv2 = ResDoubleConvBlock(in_channel=512, out_channel=256)
        self.de_up3 = UpSample(in_channel=256)
        self.de_conv3 = ResDoubleConvBlock(in_channel=256, out_channel=128)
        self.de_up4 = UpSample(in_channel=128)
        self.de_conv4 = ResDoubleConvBlock(in_channel=128, out_channel=64)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        code_lay1 = self.code_conv1(x)
        code_lay2 = self.code_conv2(self.code_down1(code_lay1))
        code_lay3 = self.code_conv3(self.code_down2(code_lay2))
        code_lay4 = self.code_conv4(self.code_down3(code_lay3))
        # 编码的最底层，也是解码的第一层
        code_lay5 = self.code_conv5(self.code_down4(code_lay4))
        deCode_lay1 = self.de_conv1(self.de_up1(code_lay5, code_lay4))
        deCode_lay2 = self.de_conv2(self.de_up2(deCode_lay1, code_lay3))
        deCode_lay3 = self.de_conv3(self.de_up3(deCode_lay2, code_lay2))
        deCode_lay4 = self.de_conv4(self.de_up4(deCode_lay3, code_lay1))
        out = self.out_conv(deCode_lay4)
        out = nn.Sigmoid()(out)
        return out


if __name__ == '__main__':
    x = torch.rand(1, 1, 256, 256)
    net = ResUnet(in_channel=1,n_classes=1)
    print(net(x).size())
