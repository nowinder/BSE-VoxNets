import torch
from torch import nn
from timm.models.layers import to_2tuple
from typing import Optional

def pair(val):
    return (val, val) if not isinstance(val, tuple) else val

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def spatial_shift1(x):
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x

def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x

class SplitAttention(nn.Module):
    def __init__(self, channel = 512, k = 3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias = False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias = False)
        self.softmax = nn.Softmax(1)
    
    def forward(self,x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)          #bs,k,n,c
        a = torch.sum(torch.sum(x_all, 1), 1)       #bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  #bs,kc
        hat_a = hat_a.reshape(b, self.k, c)         #bs,k,c
        bar_a = self.softmax(hat_a)                 #bs,k,c
        attention = bar_a.unsqueeze(-2)             # #bs,k,1,c
        out = attention * x_all                     # #bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c) # bs,h,w,c
        return out

class S2Attention(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention(channels)

    def forward(self, x):
        b, h, w, c = x.size()
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = x[:,:,:,c*2:]
        x_all = torch.stack([x1, x2, x3], 1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        return x

def ffn(dim, expansion_factor=4, dropout=0., layer = nn.Linear):
    return nn.Sequential(
                    layer(dim, dim * expansion_factor),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    layer(dim * expansion_factor, dim),
                    nn.Dropout(dropout)
                )


class ConvFFN(nn.Module):
    """Convolutional FFN Module. from https://github.com/apple/ml-fastvit/blob/main/models/fastvit.py#L348"""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0,
        ks: int = 7,
        change_chan = True
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential()
        self.conv.add_module(
            "conv",
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ks,
                padding=(ks-1)//2,
                groups=in_channels,
                bias=False,
            ),
        )
        self.conv.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.chac = change_chan


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chac: x = x.permute(0,3,1,2)
        short_cut = x
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        if not self.chac:return x+short_cut
        x = x.permute(0,2,3,1)
        return x
    
class S2Block(nn.Module):
    def __init__(self, d_model, depth, expansion_factor = 4, dropout = 0.):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, S2Attention(d_model)),
                PreNormResidual(d_model,ConvFFN(in_channels=d_model,hidden_channels=d_model*expansion_factor,ks=3,drop=dropout))
            ) for _ in range(depth)]
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.model(x)
        x = x.permute(0, 3, 1, 2)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        ),
        nn.BatchNorm2d(out_planes)
    )

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, dim, 1, 1]),requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1, 1]),requires_grad=True)

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x    

class ConvPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, init_values=1e-2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 8, 2),
                nn.GELU(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                nn.GELU(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 4:  
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim // 2, 2),
                nn.GELU(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size[0] == 2:  
            self.proj = torch.nn.Sequential(
                conv3x3(in_chans, embed_dim, 2),
                nn.GELU(),
            )
        else:
            raise("For convolutional projection, patch size has to be in [2, 4, 16]")
        self.pre_affine = Affine(in_chans)
        self.post_affine = Affine(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape 
        x = self.pre_affine(x)
        x = self.proj(x)
        x = self.post_affine(x)
        return x

class BSEVoxnet(nn.Module):
    def __init__(
        self,
        image_size=128,
        patch_size=[4, 2],
        in_channels=20,
        d_model=[192, 384],
        depth=[10, 17], # default 
        expansion_factor = [3, 3],
        ups = 3,
        drop=0.
    ):
        image_size = pair(image_size)
        oldps = [1, 1]
        for ps in patch_size:
            ps = pair(ps)
            assert (image_size[0] % (ps[0] * oldps[0])) == 0, 'image must be divisible by patch size'
            assert (image_size[1] % (ps[1] * oldps[1])) == 0, 'image must be divisible by patch size'
            oldps[0] = oldps[0] * ps[0]
            oldps[1] = oldps[1] * ps[1]
        assert (len(patch_size) == len(depth) == len(d_model) == len(expansion_factor)), 'patch_size/depth/d_model/expansion_factor must be a list'
        super().__init__()
        self.stage = len(patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                ConvPatchEmbed(img_size=image_size,patch_size=patch_size[i], in_chans= in_channels if i==0 else d_model[i-1],embed_dim=d_model[i]),
                S2Block(d_model[i], depth[i], expansion_factor[i], dropout = drop)
            ) for i in range(self.stage)]
        )
        self.upcfn = nn.Sequential(
            *[nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True),
                ConvFFN(in_channels=d_model[-1],hidden_channels=d_model[-1]*4,ks=3,drop=drop,change_chan=False)
            ) for _ in range(ups)]
        )
        self.emb2dep = nn.Conv2d(in_channels=d_model[-1],out_channels=24, kernel_size=1, bias=True)
        self.out = nn.Conv3d(in_channels=1, out_channels=2, kernel_size=5, bias=False)

    def forward(self, x):
        embedding = self.stages(x)
        convc = self.upcfn(embedding)
        dep = self.emb2dep(convc)
        dep = dep.unsqueeze(1)
        out = self.out(dep)
        return out