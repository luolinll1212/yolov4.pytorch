# -*- coding: utf-8 -*-  
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict


# Mish激活函数
class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


# 卷积块,conv+bn+mish
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x


# CSPDarkNet,内部堆叠的残差块
class ResBlock(nn.Module):
    def __init__(self, channels, hidden_channels=None, residual_activate=nn.Identity()):
        super(ResBlock, self).__init__()

        if hidden_channels is None:
            hidden_channels = channels

        self.block = nn.Sequential(
            BasicConv(channels, hidden_channels, 1),
            BasicConv(hidden_channels, channels, 3)
        )

    def forward(self, x):
        return x + self.block(x)


# CSPDarkNet的结构块,存在一个大残差边,这个大残差边绕过了很多的残差结构
class ResBlock_Body(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, first):
        super(ResBlock_Body, self).__init__()

        self.downsample_conv = BasicConv(in_channels, out_channels, 3, stride=2)

        if first:
            self.split_conv0 = BasicConv(out_channels, out_channels, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels, 1)
            self.blocks_conv = nn.Sequential(
                ResBlock(channels=out_channels, hidden_channels=out_channels // 2),
                BasicConv(out_channels, out_channels, 1)
            )
            self.concat_conv = BasicConv(out_channels * 2, out_channels, 1)
        else:
            self.split_conv0 = BasicConv(out_channels, out_channels // 2, 1)
            self.split_conv1 = BasicConv(out_channels, out_channels // 2, 1)

            self.blocks_conv = nn.Sequential(
                *[ResBlock(out_channels // 2) for _ in range(num_blocks)],
                BasicConv(out_channels // 2, out_channels // 2, 1)
            )
            self.concat_conv = BasicConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.downsample_conv(x)

        x0 = self.split_conv0(x)

        x1 = self.split_conv1(x)
        x1 = self.blocks_conv(x1)

        x = torch.cat([x1, x0], dim=1)
        x = self.concat_conv(x)

        return x


class CSPDarkNet(nn.Module):
    def __init__(self, layers):
        super(CSPDarkNet, self).__init__()
        self.inplaes = 32
        self.conv1 = BasicConv(3, self.inplaes, kernel_size=3, stride=1)
        self.feature_channels = [64, 128, 256, 512, 1024]

        self.stages = nn.ModuleList([
            ResBlock_Body(self.inplaes, self.feature_channels[0], layers[0], first=True),
            ResBlock_Body(self.feature_channels[0], self.feature_channels[1], layers[1], first=False),
            ResBlock_Body(self.feature_channels[1], self.feature_channels[2], layers[2], first=False),
            ResBlock_Body(self.feature_channels[2], self.feature_channels[3], layers[3], first=False),
            ResBlock_Body(self.feature_channels[3], self.feature_channels[4], layers[4], first=False)
        ])

        self.num_features = 1
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)

        x = self.stages[0](x)
        x = self.stages[1](x)
        out3 = self.stages[2](x)
        out4 = self.stages[3](out3)
        out5 = self.stages[4](out4)

        return out3, out4, out5


def darknet53(pretrained, **kwargs):
    model = CSPDarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model


if __name__ == '__main__':
    model = darknet53(None)
    print(model)

    x = torch.rand(1, 3, 608, 608)
    out3, out4, out5 = model(x)
    print(out3.size())
    print(out4.size())
    print(out5.size())
