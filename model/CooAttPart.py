import math

from torch import nn
import torch
from model.unet_model import DownSample, DoubleConv, UpSample, ConvBlock


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.aft_pool_conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)

        self.act = h_swish()

        self.aft_split_conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.aft_split_conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x : torch.Tensor):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.aft_pool_conv1(y)  # 削减通道数为mip
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.aft_split_conv_h(x_h).sigmoid()
        a_w = self.aft_split_conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


##=========================================================
# 本次改动：原有r 改为 16，增加Max分支
#
##========================================================
class CoordAttChange(nn.Module):
    def __init__(self, inp, oup, reduction=32,b= 1):
        super(CoordAttChange, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # mip = max(8, inp // reduction)

       # self.aft_pool_conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(1)

        #self.act = h_swish()
        # t = int(abs((math.log(inp, 2) + b) / reduction))
        # k = t if t % 2 else t + 1
        # self.act_trans_conv = nn.Conv2d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)

        self.aft_split_conv_h = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)
        self.aft_split_conv_w = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x : torch.Tensor):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = y.permute(0,3,2,1)  # trans
        y = self.bn1(y)
        _,_,_,inp = y.size()
        r = 3
        t = int(abs((math.log(inp, 2) + 1) / r))
        k = t if t % 2 else t + 1
        act_trans_conv = nn.Conv2d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False).cuda()
        y = act_trans_conv(y)
        y = y.permute(0,3,2,1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.aft_split_conv_h(x_h).sigmoid()
        a_w = self.aft_split_conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class UnetCooAtt(nn.Module):
    def __init__(self, in_channle, n_classes):
        super(UnetCooAtt, self).__init__()
        self.n_channels = in_channle
        self.n_classes = n_classes
        # self.att = Res_Att_Conv_Block()
        self.c1 = ConvBlock(in_channle, 64)
        self.c1_att = CoordAttChange(64, 64)
        self.d1 = DownSample()
        self.c2 = ConvBlock(64, 128)
        self.c2_att = CoordAttChange(128, 128)  # 每次下采样的初始通道数
        self.d2 = DownSample()
        self.c3 = ConvBlock(128, 256)
        self.c3_att = CoordAttChange(256, 256)
        self.d3 = DownSample()
        self.c4 = ConvBlock(256, 512)
        self.c4_att = CoordAttChange(512, 512)
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
    x = torch.rand(1,64,64,64)
    net = UnetCooAtt(64,64)
    t = net(x)
    print(t.size())
