import torch
from torch import nn, Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def basic_convblock(inc, outc):
    return nn.Sequential(
        nn.Conv2d(inc, outc, kernel_size=3, stride=1, padding=1), 
        nn.BatchNorm2d(outc), 
        nn.ReLU(inplace=True))

class ResBlock(nn.Module):
    def __init__(self, inoutc: int, stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.conv1 = basic_convblock(inc=inoutc, outc=inoutc)
        self.conv2 = nn.Conv2d(inoutc, inoutc, kernel_size=3, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(inoutc)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:

        identity = self.conv1(x)
        out = self.conv1(identity)
        out = self.conv2(out)
        out += identity
        out = self.bn(out)
        out = self.relu(out)
        
        return out

class ResUnet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c = [3,64,128,256,512]

        self.conv_e1 = basic_convblock(inc=c[0], outc=c[1])
        self.conv_e2 = basic_convblock(inc=c[1], outc=c[1])
        self.conv_e3 = basic_convblock(inc=c[1], outc=c[2])
        self.conv_e4 = basic_convblock(inc=c[2], outc=c[3])
        self.conv_e5 = basic_convblock(inc=c[3], outc=c[4])
        self.conv_e6 = basic_convblock(inc=c[4], outc=c[4])

        self.skip1 = basic_convblock(inc=c[0], outc=c[1])
        self.skip23 = basic_convblock(inc=c[1], outc=c[1])
        self.skip4 = basic_convblock(inc=c[2], outc=c[2])
        self.skip5 = basic_convblock(inc=c[3], outc=c[3])
        self.skip67 = basic_convblock(inc=c[4], outc=c[4])

        self.res_e12_d1 = ResBlock(inoutc=c[1])
        self.res_e3_d2 = ResBlock(inoutc=c[2])
        self.res_e4_d3 = ResBlock(inoutc=c[3])
        self.res_e5_d54 = ResBlock(inoutc=c[4])

        self.conv_d1 = basic_convblock(inc=c[1]*2, outc=c[0])
        self.conv_d2 = basic_convblock(inc=c[1], outc=c[1])
        self.conv_d3 = basic_convblock(inc=c[1]*2, outc=c[1])
        self.conv_d4 = basic_convblock(inc=c[1]+c[2], outc=c[1])
        self.conv_d5 = basic_convblock(inc=c[2]+c[3], outc=c[2])
        self.conv_d6 = basic_convblock(inc=c[3]+c[4], outc=c[3])
        self.conv_d7 = basic_convblock(inc=c[4]*2, outc=c[4])

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        self.upsample = nn.Upsample(size=None, scale_factor=2, mode='nearest', align_corners=None, recompute_scale_factor=None)

    def forward(self,x):
        out = self.conv_e1(x)
        e1 = self.res_e12_d1(out)
        out = self.maxpool(e1)

        out = self.conv_e2(out)
        e2 = self.res_e12_d1(out)
        out = self.maxpool(e2)

        out = self.conv_e3(out)
        e3 = self.res_e3_d2(out)
        out = self.maxpool(e3)

        out = self.conv_e4(out)
        e4 = self.res_e4_d3(out)
        out = self.maxpool(e4)

        out = self.conv_e5(out)
        e5 = self.res_e5_d54(out)
        out = self.maxpool(e5)

        e6 = self.conv_e6(out)
        s1 = self.skip67(e6)
        out = self.res_e5_d54(s1)
        out = self.upsample(out)

        out = torch.cat([out, self.skip67(e5)], dim=1)
        out = self.conv_d7(out)
        out = self.res_e5_d54(out)
        out = self.upsample(out)

        out = torch.cat([out, self.skip5(e4)], dim=1)
        out = self.conv_d6(out)
        out = self.res_e4_d3(out)
        out = self.upsample(out)

        out = torch.cat([out, self.skip4(e3)], dim=1)
        out = self.conv_d5(out)
        out = self.res_e3_d2(out)
        out = self.upsample(out)

        out = torch.cat([out, self.skip23(e2)], dim=1)
        out = self.conv_d4(out)
        out = self.res_e12_d1(out)
        out = self.upsample(out)
        
        out = torch.cat([out, self.skip23(e1)], dim=1)
        out = self.conv_d3(out)
        out = self.conv_d2(out)
        out = torch.cat([out, self.skip1(x)], dim=1)
        out = self.conv_d1(out)

        return out

