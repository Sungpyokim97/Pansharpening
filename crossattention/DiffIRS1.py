import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
from MSDCNN.MSDCNN import MSDCNN
import mspanmixer as mixer
import math

class DiffIRS1(nn.Module):
    def __init__(self, 
        n_encoder_res=4,         
        ms_channels=8,
        pan_channels=1, 
        dim = 16,
        # num_blocks = [4,6,6,8],
        num_blocks = [2,1,1,1],
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        ):
        super(DiffIRS1, self).__init__()

        # Generator
        self.G = DIRformer(        
        ms_channels=ms_channels,
        pan_channels=pan_channels,
        dim = dim,
        num_blocks = num_blocks, 
        num_refinement_blocks = num_refinement_blocks,
        heads = heads,
        ffn_expansion_factor = ffn_expansion_factor,
        bias = bias,
        LayerNorm_type = LayerNorm_type,   ## Other option 'BiasFree'
        )

        self.E = CPEN(channels=ms_channels, n_feats=64, n_encoder_res=n_encoder_res)

        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.training = False


    def forward(self, ms, pan, gt):
        if self.training:

            IPRS1, s_z, S1_IPR = self.E(ms, pan, gt)

            sr = self.G(ms, pan, IPRS1)

            return sr, S1_IPR
        else:
            IPRS1, s_z, S1_IPR = self.E(ms, pan, gt)
            sr = self.G(ms, pan, IPRS1)
            return sr, S1_IPR
        
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor) # 48*2.66 = 128

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )   #256 -> 96
    def forward(self, x,k_v):
        b,c,h,w = x.shape
        k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=1)
        x = x*k_v1+k_v2  
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
    

#modified for channel swin
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x,k_v):
        b,c,h,w = x.shape
        k_v=self.kernel(k_v).view(-1,c*2,1,1)
        k_v1,k_v2=k_v.chunk(2, dim=1)
        
        x = x*k_v1+k_v2


        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = (q.transpose(-2, -1) @ k)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        # out = (v @ attn)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]
        k_v=y[1]
        x = x + self.ffn(self.norm2(x),k_v)
        x = x + self.attn(self.norm1(x),k_v)

        return [x,k_v]


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2),
                                     nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True))
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        attention = torch.sigmoid(x_out) # broadcasting
        return attention

# modified by sungpyo 
class CPEN(nn.Module):
    def __init__(self, channels = 8,  n_feats = 64, n_encoder_res = 6, scale=4):
        super(CPEN, self).__init__()
        self.scale=scale
        E1=[nn.Conv2d(channels*16 + channels*16 + 1*16, n_feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True)]
        E2=[
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]

        E3=[
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1),      # 64 128
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1),  # 128 128
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1),  # 128 256
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        ]
        E_12 = E1 + E2
        self.E_12 = nn.Sequential(
            *E_12
        )
        self.E3 = nn.Sequential(
            *E3
        ) 
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(n_feats * 4, n_feats * 4),
        )
        self.spatial = SpatialGate()
        self.pixel_unshuffle = nn.PixelUnshuffle(4)
        self.pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pixel_unshufflev2 = nn.PixelUnshuffle(2)
        self.upsample = nn.Sequential(nn.Conv2d(3, 3*(4), kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.LeakyReLU(0.1, True),
                                      nn.Conv2d(3*(4), 3*(16), kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.PixelShuffle(4))
    def forward(self, ms, pan, gt):
        gt0 = self.pixel_unshuffle(gt)  # (c*r^2) h w 
        S1_IPR = []
        # gt0 = gt
        # feat = self.upsample(x)
        ms = self.pixel_unshuffle(ms)   # (c*r^2) h w 
        pan = self.pixel_unshuffle(pan) # (r^2) h w
        x = torch.cat([ms, pan, gt0], dim=1)   # 6 256 256
        mid = self.E_12(x)
        # channel z
        fea1 = self.E3(mid).squeeze(-1).squeeze(-1)
        c_z = torch.sigmoid(self.mlp(fea1))
        S1_IPR.append(c_z)
        # spatial z
        fea2 = self.pool(mid)
        s_z = self.spatial(fea2)
        S1_IPR.append(s_z)
        return c_z, s_z, S1_IPR


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
    
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)
    
class ResBlock(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act=nn.ReLU(), res_scale=1):
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

class Upsampler(nn.Module):
    def __init__(self, conv, scale, n_feats, act=False):
        super(Upsampler, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feats, scale ** 2 * n_feats, 3, padding=1),
            nn.PixelShuffle(scale)
        )
        if act:
            self.body = nn.Sequential(self.body, nn.ReLU(True))

    def forward(self, x):
        x = self.body(x)
        return x
    
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

    
class DIRformer(nn.Module):
    def __init__(self, 
        ms_channels=8,
        pan_channels=1, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):

        super(DIRformer, self).__init__()

        self.mixer = MSDCNN(ms_channels)
        # self.mixer = mixer.PGCU(ms_channels, Vec = 256)
        self.channel_mixer = mixer.Channel_mixer(ms_channels)
        
        # self.layernorm = nn.LayerNorm()
        self.patch_embed = OverlapPatchEmbed(in_c=ms_channels, embed_dim=dim)
        
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        self.output = nn.Conv2d(int(dim*2**1), ms_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu = nn.ReLU()
    def forward(self, ms, pan ,k_v):

        ms_b, ms_c, ms_h, ms_w = ms.shape
        pan_b, pan_c, pan_h, pan_w = pan.shape
        # exp_list = [0]*ms_c
        # pan_expand = pan[:,exp_list,...]
        # ms_pan_simplemixed = self.mixer(pan, ms) + self.channel_mixer(pan, ms)[0]
        ms_pan_simplemixed = ms
        inp_enc_level1 = self.patch_embed(ms_pan_simplemixed)

        out_enc_level1,_ = self.encoder_level1([inp_enc_level1,k_v])
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2, _ = self.encoder_level2([inp_enc_level2,k_v])

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3,_ = self.encoder_level3([inp_enc_level3,k_v]) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent,_ = self.latent([inp_enc_level4,k_v]) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3,_ = self.decoder_level3([inp_dec_level3,k_v]) 

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2,_ = self.decoder_level2([inp_dec_level2,k_v]) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1,_ = self.decoder_level1([inp_dec_level1,k_v])
        
        out_dec_level1,_ = self.refinement([out_dec_level1,k_v])

        out_dec_level1 = self.output(out_dec_level1) + ms_pan_simplemixed
        out = self.relu(out_dec_level1)
        return out

if __name__ == '__main__':
    h, w = [64, 64]
    ms = torch.rand(5, 8, h*4, w*4)
    pan = torch.rand(5, 1, h*4, w*4)
    gt = torch.rand(5, 8, h*4, w*4)
    feat = torch.rand(5, 64, 64, 64)
    cpen = CPEN(n_feats = 64, n_encoder_res = 6)
    model = DiffIRS1()
    # c_z, s_z, ipr = cpen(ms, pan, gt)
    out = model(ms, pan, gt)
    # print(out.shape)

    
    
    
