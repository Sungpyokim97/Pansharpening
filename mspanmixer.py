import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import math


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act=nn.LeakyReLU(), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for _ in range(2):
            m.append(conv(n_feats, n_feats, kernel_size))
            if act is not None:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

#simple only concat
class Simplemixer(nn.Module):
    def __init__(self, ms_channels, pan_channels):
        super(Simplemixer, self).__init__()

        self.res = ResBlock(
                default_conv, n_feats = ms_channels+pan_channels, kernel_size=3
            )
        self.conv = default_conv(ms_channels+pan_channels,ms_channels, kernel_size=3)

    def forward(self, ms, pan):
        x = torch.cat((ms, pan), dim=1)
        x = self.res(x)
        x = self.conv(x)

        return x


if __name__ == '__main__':
    h, w = [64, 64]
    ms = torch.rand(5, 8, h*4, w*4)
    pan = torch.rand(5, 1, h*4, w*4)
    gt = torch.rand(5, 8, h*4, w*4)
    feat = torch.rand(5, 64, 64, 64)
    simple = Simplemixer(8,1)
    out = simple(ms, pan)
    print(out.shape)
#complex
