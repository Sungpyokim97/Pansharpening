import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from functools import partial
import tqdm
from inspect import isfunction
from CA_S1 import DIRformer

class DiffIRS2(nn.Module):
    def __init__(self,         
        n_encoder_res=4,         
        ms_channels=8,
        dim = 16,
        # num_blocks = [4,6,6,8],
        num_blocks = [2,1,1,1],
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        ## diffusion process
        n_denoise_res = 1, 
        linear_start= 0.1,
        linear_end= 0.99, 
        timesteps = 10):
        super(DiffIRS2, self).__init__()

        # Generator
        self.G = DIRformer(        
        ms_channels=ms_channels,
        dim = dim,
        num_blocks = num_blocks, 
        num_refinement_blocks = num_refinement_blocks,
        heads = heads,
        ffn_expansion_factor = ffn_expansion_factor,
        bias = bias,
        LayerNorm_type = LayerNorm_type,   ## Other option 'BiasFree'
        )
        self.condition = CPEN2(channels = ms_channels, n_feats=64, n_encoder_res=n_encoder_res)

        self.denoise_cz= denoise(n_feats=64, 
                                 n_denoise_res=n_denoise_res,
                                 timesteps=timesteps)
        
        self.diffusion_cz = DDPM(denoise=self.denoise_cz, 
                                 condition=self.condition,
                                 n_feats=64,
                                 linear_start= linear_start,
                                 linear_end= linear_end, 
                                 timesteps = timesteps)

    def forward(self, ms, pan, IPRS1=None):
        if self.training:
            IPRS2, pred_IPR_list=self.diffusion_cz(ms, pan, IPRS1)
            sr = self.G(ms, pan, IPRS2)
            return sr, pred_IPR_list
        else:
            IPRS2 = self.diffusion_cz(ms, pan)
            sr = self.G(ms, pan, IPRS2)
            return sr

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )
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

class CPEN2(nn.Module):
    def __init__(self, channels = 8, n_feats = 64, n_encoder_res = 6):
        super(CPEN2, self).__init__()
 
        # scale == 4    in worldview3, 257ch -> 64ch
        E1=[nn.Conv2d(channels*16 + 1*16, n_feats, kernel_size=3, padding=1),
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
        
    def forward(self, ms, pan):
        S2_IPR = []
        # gt0 = gt
        # feat = self.upsample(x)
        ms = self.pixel_unshuffle(ms)   # (c*r^2) h w 
        pan = self.pixel_unshuffle(pan) # (r^2) h w
        x = torch.cat([ms, pan], dim=1)   # 6 256 256
        mid = self.E_12(x)
        # channel z
        fea1 = self.E3(mid).squeeze(-1).squeeze(-1)
        c_z = torch.sigmoid(self.mlp(fea1))
        S2_IPR.append(c_z)
        # spatial z
        fea2 = self.pool(mid)
        s_z = self.spatial(fea2)
        S2_IPR.append(s_z)
        return c_z, s_z, S2_IPR

class ResMLP(nn.Module):
    def __init__(self,n_feats = 512):
        super(ResMLP, self).__init__()
        self.resmlp = nn.Sequential(
            nn.Linear(n_feats , n_feats ),
            nn.LeakyReLU(0.1, True),
        )
    def forward(self, x):
        res=self.resmlp(x)
        return res

class denoise(nn.Module):
    def __init__(self,n_feats = 64, n_denoise_res = 5,timesteps=5):
        super(denoise, self).__init__()
        self.max_period=timesteps*10
        n_featsx4=4*n_feats
        resmlp = [
            nn.Linear(n_featsx4*2+1, n_featsx4),
            nn.LeakyReLU(0.1, True),
        ]
        for _ in range(n_denoise_res):
            resmlp.append(ResMLP(n_featsx4))
        self.resmlp=nn.Sequential(*resmlp)

    def forward(self,x, t,c):
        t=t.float()
        t =t/self.max_period
        t=t.view(-1,1)
        c = torch.cat([c,t,x],dim=1)
        
        fea = self.resmlp(c)

        return fea 
        
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.LeakyReLU(0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res
    
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, act=False, bias=True):
        m = []
        if (int(scale) & (int(scale) - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if act: m.append(act())
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2

class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 denoise,
                 condition,
                 timesteps=1000,
                 beta_schedule="linear",
                 image_size=256,
                 n_feats=128,
                 clip_denoised=False,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 parameterization="x0",  # all assuming fixed variance schedules
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        # self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.image_size = image_size  # try conv?
        self.channels = n_feats
        self.model = denoise
        self.condition = condition

        self.v_posterior = v_posterior
        self.l_simple_weight = l_simple_weight

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, c,clip_denoised: bool):
        model_out = self.model(x, t, c)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, model_out

    def p_sample(self, x, t, c,clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, predicted_noise = self.p_mean_variance(x=x, t=t, c=c,clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean , predicted_noise

    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        # loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])


        return model_out, target

    def forward(self, ms, pan ,x=None):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        device = self.betas.device
        b=ms.shape[0]
        if self.training:
            pred_IPR_list=[]
            t= torch.full((b,), self.num_timesteps-1,  device=device, dtype=torch.long)
            noise = torch.randn_like(x)
            x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
            IPR = x_noisy

            c = self.condition(ms, pan)
            # 추가
            c = c[0]
            for i in reversed(range(0, self.num_timesteps)):
                IPR, predicted_noise = self.p_sample(IPR, torch.full((b,), i,  device=device, dtype=torch.long), c,
                                clip_denoised=self.clip_denoised)
                pred_IPR_list.append(IPR)
            return IPR,pred_IPR_list
        else:       
            shape=(ms.shape[0],self.channels*4)
            x_noisy = torch.randn(shape, device=device)
            c = self.condition(ms, pan)

            c = c[0]
            IPR = x_noisy
            for i in reversed(range(0, self.num_timesteps)):
                IPR, _ = self.p_sample(IPR, torch.full((b,), i,  device=device, dtype=torch.long), c,
                                clip_denoised=self.clip_denoised)
            return IPR
        
def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts

def exists(x):
    return x is not None

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.optim import Adam
    import os
    from tqdm import tqdm
    import configparser
    import numpy as np
    from torch.utils.tensorboard import SummaryWriter

    from DiffIRS1 import DiffIRS1
    from DiffIRS2 import DiffIRS2
    from torchvision.utils import save_image
    from torch.nn import DataParallel

    # # stage2에 들어오기 전 stage1을 거치기 위함
    # device = 'cuda'
    # pre_model = DiffIRS1().to(device)
    # model = DiffIRS2().to(device)

    # # stage1에 내가 원하는 weight를 씌워서 CPEN 작동시킬 준비
    # weight_dir = '/home/ksp/Desktop/Diff/ckpt/test/checkpoint_epoch_191_EDSR.pth'
    # checkpoint = torch.load(weight_dir)
    # # print(checkpoint['model_state_dict'].keys())
    # state_dict = checkpoint['model_state_dict']
    # # 'module.G'에 해당하는 키들을 필터링
    # new_state_dict = {}
    # for (key,val) in state_dict.items():
    #     if key.startswith('module.G'):
    #         # print(type(key))
    #         key = key.replace('module.G.', '')
    #         new_state_dict[key]= val

    # print(new_state_dict.keys())
    # # module_G_keys = [key for key in state_dict.keys() if key.startswith('module.G')]

    # # # 'module.G'에 속한 가중치들만 추출
    # # module_G_weights = {key: state_dict[key] for key in module_G_keys}
    # # # 주어진 dict_keys

    # # # "module.G." 부분을 제거한 새로운 키 리스트 생성
    # # new_keys = [module_G_weights.key.replace('module.G.', '') for key in module_G_keys]

    # # 새로운 키 리스트 출력
    # # print(new_keys)
    # # print(module_G_weights.keys())
    # # S1
    # # pre_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # # model.G = pre_model.G


    condition = CPEN2(n_feats=64, n_encoder_res=4)

    denoise_cz= denoise(n_feats=64, 
                                n_denoise_res=1,
                                timesteps=4)
    
    diffusion_cz = DDPM(denoise=denoise_cz, 
                                condition=condition,
                                n_feats=64,
                                linear_start= 0.1,
                                linear_end= 0.99, 
                                timesteps = 50)
    h,w = 64,64
    ms = torch.rand(1,8,h*4,w*4)   # B C H W
    pan = torch.rand(1,1,h*4,w*4)
    IPRS1 = torch.rand(1,256)     # B C 

    ipr, pred_ipr = diffusion_cz(ms, pan, IPRS1)