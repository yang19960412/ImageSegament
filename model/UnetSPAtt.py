from torch import nn
import torch
from model.unet_model import ConvBlock, DownSample, UpSample


class SpAtt(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpAtt, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        out = x * out
        return out


class UnetSpAtt(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UnetSpAtt, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # self.att = Res_Att_Conv_Block()
        self.c1 = ConvBlock(n_channels, 64)
        self.c1_att = SpAtt()
        self.d1 = DownSample()
        self.c2 = ConvBlock(64, 128)
        self.c2_att = SpAtt()  # 每次下采样的初始通道数
        self.d2 = DownSample()
        self.c3 = ConvBlock(128, 256)
        self.c3_att = SpAtt()
        self.d3 = DownSample()
        self.c4 = ConvBlock(256, 512)
        self.c4_att = SpAtt()
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


if __name__ == '__main__':
    x = torch.rand(1, 1, 256, 256)
    net = UnetSpAtt(1, 1)
    print(net(x).size())
