import torch
import torch.nn as nn
import torch.nn.functional as func
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
        for _ in range(8):
            m.append(conv(n_feats, n_feats, kernel_size))
            if act is not None:
                m.append(act)
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res
    
class DownSamplingBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel):
        super(DownSamplingBlock, self).__init__()
        self.Conv = nn.Conv2d(in_channel, out_channel, (3,3), 2, 1)
        self.MaxPooling = nn.MaxPool2d((2, 2))
        
    def forward(self, x):
        out = self.MaxPooling(self.Conv(x))
        return out
    
class PGCU(nn.Module):
    
    def __init__(self, Channel=4, Vec=128, NumberBlocks=3):
        super(PGCU, self).__init__()
        self.BandVecLen = Vec//Channel
        self.Channel = Channel
        self.VecLen = Vec
        
        ## Information Extraction
        # F.size == (Vec, W, H)
        self.FPConv = nn.Conv2d(1, Channel, (3,3), 1, 1)
        self.FMConv = nn.Conv2d(Channel, Channel, (3,3), 1, 1)
        self.FConv = nn.Conv2d(Channel*2, Vec, (3,3), 1, 1)
        # G.size == (Vec, W/pow(2, N), H/pow(2, N))
        self.GPConv = nn.Sequential()
        self.GMConv = nn.Sequential()
        self.GConv = nn.Conv2d(Channel*2, Vec, (3,3), 1, 1)
        for i in range(NumberBlocks):
            if i == 0:
                self.GPConv.add_module('DSBlock'+str(i), DownSamplingBlock(1, Channel))
            else:
                self.GPConv.add_module('DSBlock'+str(i), DownSamplingBlock(Channel, Channel))
                self.GMConv.add_module('DSBlock'+str(i-1), DownSamplingBlock(Channel, Channel))
        # V.size == (C, W/pow(2, N), H/pow(2, N)), k=W*H/64
        self.VPConv = nn.Sequential()
        self.VMConv = nn.Sequential()
        self.VConv = nn.Conv2d(Channel*2, Channel, (3,3), 1, 1)
        for i in range(NumberBlocks):
            if i == 0:
                self.VPConv.add_module('DSBlock'+str(i), DownSamplingBlock(1, Channel))
            else:
                self.VPConv.add_module('DSBlock'+str(i), DownSamplingBlock(Channel, Channel))
                self.VMConv.add_module('DSBlock'+str(i-1), DownSamplingBlock(Channel, Channel))

        # Linear Projection
        self.FLinear = nn.ModuleList([nn.Sequential(nn.Linear(self.VecLen, self.BandVecLen), nn.LayerNorm(self.BandVecLen)) for i in range(self.Channel)])
        self.GLinear = nn.ModuleList([nn.Sequential(nn.Linear(self.VecLen, self.BandVecLen), nn.LayerNorm(self.BandVecLen)) for i in range(self.Channel)])
        # FineAdjust
        self.FineAdjust = nn.Conv2d(Channel, Channel, (3,3), 1, 1)
        
    def forward(self, guide, x):

        if x.shape[2:] == guide.shape[2:]:
            up_x = x
            x = func.interpolate(x, scale_factor=(0.25, 0.25), mode='nearest')
        else:
            up_x = func.interpolate(x, scale_factor=(4,4), mode='nearest')
        Fm = self.FMConv(up_x)
        Fq = self.FPConv(guide)
        F = self.FConv(torch.cat([Fm, Fq], dim=1))
        
        Gm = self.GMConv(x)
        Gp = self.GPConv(guide)
        G = self.GConv(torch.cat([Gm, Gp], dim=1))
        
        Vm = self.VMConv(x)
        Vp = self.VPConv(guide)
        V = self.VConv(torch.cat([Vm, Vp], dim=1))
        
        C = V.shape[1]
        batch = G.shape[0]
        W, H = F.shape[2], F.shape[3]
        OW, OH = G.shape[2], G.shape[3]
        
        G = torch.transpose(torch.transpose(G, 1, 2), 2, 3)
        G = G.reshape(batch*OW*OH, self.VecLen)
        
        F = torch.transpose(torch.transpose(F, 1, 2), 2, 3)
        F = F.reshape(batch*W*H, self.VecLen)
        BandsProbability = None
        for i in range(C):
            # F projection
            FVF = self.GLinear[i](G)
            FVF = FVF.reshape(batch, OW*OH, self.BandVecLen).transpose(-1, -2) # (batch, L, OW*OH)
            # G projection
            PVF = self.FLinear[i](F)
            PVF = PVF.view(batch, W*H, self.BandVecLen) # (batch, W*H, L)
            # Probability
            Probability = torch.bmm(PVF, FVF).reshape(batch*H*W, OW, OH) / math.sqrt(self.BandVecLen)
            Probability = torch.exp(Probability) / torch.sum(torch.exp(Probability), dim=(-1, -2)).unsqueeze(-1).unsqueeze(-1)
            Probability = Probability.view(batch, W, H, 1, OW, OH)
            # Merge
            if BandsProbability is None:
                BandsProbability = Probability
            else:
                BandsProbability = torch.cat([BandsProbability, Probability], dim=3)
        #Information Entropy: H_map = torch.sum(BandsProbability*torch.log2(BandsProbability+1e-9), dim=(-1, -2, -3)) / C
        out = torch.sum(BandsProbability*V.unsqueeze(dim=1).unsqueeze(dim=1), dim=(-1, -2))
        out = out.transpose(-1, -2).transpose(1, 2)
        out = self.FineAdjust(out)
        return out

#simple only concat
class Simple_mixer(nn.Module):
    def __init__(self, ms_channels):
        super(Simple_mixer, self).__init__()

        self.res = ResBlock(
                default_conv, n_feats = ms_channels*2, kernel_size=3
            )
        self.conv = default_conv(ms_channels*2,ms_channels, kernel_size=3)
        # self.conv1x1 = nn.Conv2d()

    def forward(self, pan, ms):
        if pan.shape[1] == 1:
            pan_expand = pan[:,[0]*ms.shape[1],...]
        else:
            pan_expand = pan
        x = torch.cat((ms, pan_expand), dim=1)
        x = self.res(x)
        x = self.conv(x)

        return x
class Channel_mixer(nn.Module):
    def __init__(self, ms_channels):
        super(Channel_mixer, self).__init__()
        self.linear_pan = nn.Sequential(
            nn.Linear(ms_channels * 4, ms_channels * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(ms_channels * 4, ms_channels),
        )
        self.linear_ms = nn.Sequential(
            nn.Linear(ms_channels * 4, ms_channels * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(ms_channels * 4, ms_channels),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.AdaptiveMaxPool2d((1,1))
    def forward(self, pan, ms):
        if pan.shape[1] == 1:
            pan_extend = pan[:,[0]*ms.shape[1],...]
        else:
            pan_extend = pan
        pan_concat = torch.cat((self.maxpool(pan_extend), self.avgpool(pan_extend)), dim=1)
        ms_concat = torch.cat((self.maxpool(ms), self.avgpool(ms)), dim=1)
        concat = torch.cat((pan_concat, ms_concat), dim=1).squeeze(-1).squeeze(-1)
        ccpan = rearrange(self.linear_pan(concat), 'b c -> b c 1 1')
        ccms = rearrange(self.linear_ms(concat), 'b c -> b c 1 1')
        panout = pan*ccpan + pan
        msout = ms*ccms + ms
        return panout, msout
    
class PreMixer(nn.Module):
    def __init__(self, spectral_num, criterion=None, channel=64):
        super(PreMixer, self).__init__()

        self.criterion = criterion

        input_channel = spectral_num + 1
        output_channel = spectral_num

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=60, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv2_1 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv3 = nn.Conv2d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4_3 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv5 = nn.Conv2d(in_channels=30, out_channels=output_channel, kernel_size=5, stride=1, padding=2, bias=True)

        self.shallow1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
        self.shallow2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        self.shallow3 = nn.Conv2d(in_channels=32, out_channels=output_channel, kernel_size=5, stride=1, padding=2, bias=True)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x, y):  # x: lms; y: pan

        concat = torch.cat([x, y], 1)  # Bsx9x64x64

        out1 = self.relu(self.conv1(concat))  # Bsx60x64x64
        out21 = self.conv2_1(out1)   # Bsx20x64x64
        out22 = self.conv2_2(out1)   # Bsx20x64x64
        out23 = self.conv2_3(out1)   # Bsx20x64x64
        out2 = torch.cat([out21, out22, out23], 1)  # Bsx60x64x64

        out2 = self.relu(torch.add(out2, out1))  # Bsx60x64x64

        out3 = self.relu(self.conv3(out2))  # Bsx30x64x64
        out41 = self.conv4_1(out3)          # Bsx10x64x64
        out42 = self.conv4_2(out3)          # Bsx10x64x64
        out43 = self.conv4_3(out3)          # Bsx10x64x64
        out4 = torch.cat([out41, out42, out43], 1)  # Bsx30x64x64

        out4 = self.relu(torch.add(out4, out3))  # Bsx30x64x64

        out5 = self.conv5(out4)  # Bsx8x64x64

        shallow1 = self.relu(self.shallow1(concat))   # Bsx64x64x64
        shallow2 = self.relu(self.shallow2(shallow1))  # Bsx32x64x64
        shallow3 = self.shallow3(shallow2) # Bsx8x64x64

        out = out5+ shallow3+ y  # Bsx8x64x64
        out = self.relu(out)  # Bsx8x64x64

        return out
if __name__ == '__main__':
    h, w = [32, 32]
    ms = torch.rand(5, 8, h*4, w*4)
    pan = torch.rand(5, 1, h*4, w*4)
    gt = torch.rand(5, 8, h*4, w*4)
    feat = torch.rand(5, 64, 64, 64)
    simple = Channel_mixer(8)
    pgcu = PGCU(Channel=8, Vec=128)
    out_simple = simple(pan, ms)
    out_pgcu = pgcu(pan, ms)
    print(out_simple[0].shape, out_pgcu.shape)
#complex
