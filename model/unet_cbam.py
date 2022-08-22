from torch import  nn
from model.unet_model import ConvBlock
from model.cbam import CBAMBlock
from model.unet_model import DownSample,UpSample

class UnetCBAM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UnetCBAM, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        #self.att = Res_Att_Conv_Block()
        self.c1 = ConvBlock(n_channels, 64)
        self.c1_att = CBAMBlock(64)
        self.d1 = DownSample()
        self.c2 = ConvBlock(64, 128)
        self.c2_att = CBAMBlock(128)  #每次下采样的初始通道数
        self.d2 = DownSample()
        self.c3 = ConvBlock(128, 256)
        self.c3_att = CBAMBlock(256)
        self.d3 = DownSample()
        self.c4 = ConvBlock(256, 512)
        self.c4_att = CBAMBlock(512)
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
       # code_lay1_att = self.c1_att(code_lay1)
        code_lay2 = self.c2(self.d1(code_lay1))
       # code_lay2_att = self.c2_att(code_lay2)
        code_lay3 = self.c3(self.d2(code_lay2))
       # code_lay3_att = self.c3_att(code_lay3)
        code_lay4 = self.c4(self.d3(code_lay3))
        code_lay4_att = self.c4_att(code_lay4)
        #编码的最底层，也是解码的第一层
        code_lay5 = self.c5(self.d4(code_lay4))
        deCode_lay1 = self.c6(self.u1(code_lay5, code_lay4_att))
        deCode_lay2 = self.c7(self.u2(deCode_lay1, code_lay3))
        deCode_lay3 = self.c8(self.u3(deCode_lay2, code_lay2))
        deCode_lay4 = self.c9(self.u4(deCode_lay3, code_lay1))
        out = self.out(deCode_lay4)
        out = nn.Sigmoid()(out)
        return out
    def getName(self):
        return str(UnetCBAM);