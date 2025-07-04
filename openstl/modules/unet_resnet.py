from functools import partial
from typing import Optional

import torch
from einops import rearrange
from torch import nn

from .attention_modules import Attention, LinearAttention
from .custom_norm_layers import PreNorm, WeightStandardizedConv2d
from .misc_modules import Residual, get_time_embedder


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, dropout: float = 0.0):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        x = self.dropout(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        groups=8,
        double_conv_layer: bool = True,
        dropout1: float = 0.0,
        dropout2: float = 0.0,
    ):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups, dropout=dropout1)
        self.block2 = (
            Block(dim_out, dim_out, groups=groups, dropout=dropout2)
            if double_conv_layer
            else nn.Identity()
        )
        self.residual_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.residual_conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        dim,
        channels,  # Channels of the primary input x_t
        out_channels: Optional[int] = None,
        self_condition: bool = False,
        condition_channels: Optional[int] = None,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        resnet_block_groups: int = 8,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        time_embed_dim_ratio: float = 4.0,
        attention_resolutions: tuple[int] = (1,),
        attention_head_dim: int = 64,
        attention_heads: Optional[int] = None,
        dropout_resnets: tuple[float, ...] = (0.0,),
        dropout_attns: tuple[float, ...] = (0.0,),
        double_conv_layer_down: tuple[bool, ...] = (True,),
        double_conv_layer_up: tuple[bool, ...] = (True,),
        num_classes: Optional[int] = None,
        **kwargs,  # Accept and ignore extra arguments
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        self.condition_channels = condition_channels or 0
        out_channels = default(out_channels, channels)

        # Calculate the total number of input channels for the initial convolution
        init_conv_channels = channels
        if self.self_condition:
            init_conv_channels += channels  # Self-conditioning tensor has same channels as input
        if self.condition_channels > 0:
            init_conv_channels += self.condition_channels

        self.init_conv = nn.Conv2d(init_conv_channels, dim, 1, padding=0)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = int(dim * time_embed_dim_ratio)
        self.time_mlp = get_time_embedder(
            time_dim,
            learned_sinusoidal_dim,
            random_fourier_features,
            learned_sinusoidal_cond,
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        if len(dropout_resnets) < num_resolutions:
            dropout_resnets = dropout_resnets + (dropout_resnets[-1],) * (
                num_resolutions - len(dropout_resnets)
            )
        if len(dropout_attns) < num_resolutions:
            dropout_attns = dropout_attns + (dropout_attns[-1],) * (
                num_resolutions - len(dropout_attns)
            )
        if len(double_conv_layer_down) < num_resolutions:
            double_conv_layer_down = double_conv_layer_down + (double_conv_layer_down[-1],) * (
                num_resolutions - len(double_conv_layer_down)
            )
        if len(double_conv_layer_up) < num_resolutions:
            double_conv_layer_up = double_conv_layer_up + (double_conv_layer_up[-1],) * (
                num_resolutions - len(double_conv_layer_up)
            )

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            use_attn = num_resolutions - 1 - ind in attention_resolutions
            if attention_heads is None:
                attn_heads = dim_in // attention_head_dim
            else:
                attn_heads = attention_heads

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            double_conv_layer=double_conv_layer_down[ind],
                            dropout1=dropout_resnets[ind],
                            dropout2=dropout_resnets[ind],
                        ),
                        block_klass(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            double_conv_layer=double_conv_layer_down[ind],
                            dropout1=dropout_resnets[ind],
                            dropout2=dropout_resnets[ind],
                        ),
                        Residual(
                            PreNorm(
                                dim_in,
                                LinearAttention(
                                    dim_in, heads=attn_heads, dim_head=attention_head_dim
                                ),
                            )
                        )
                        if use_attn
                        else nn.Identity(),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(
            PreNorm(mid_dim, Attention(mid_dim, dim_head=attention_head_dim, heads=attn_heads))
        )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            use_attn = num_resolutions - 1 - ind in attention_resolutions
            if attention_heads is None:
                attn_heads = dim_out // attention_head_dim
            else:
                attn_heads = attention_heads

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            double_conv_layer=double_conv_layer_up[ind],
                            dropout1=dropout_resnets[ind],
                            dropout2=dropout_resnets[ind],
                        ),
                        block_klass(
                            dim_out + dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            double_conv_layer=double_conv_layer_up[ind],
                            dropout1=dropout_resnets[ind],
                            dropout2=dropout_resnets[ind],
                        ),
                        Residual(
                            PreNorm(
                                dim_out,
                                LinearAttention(
                                    dim_out, heads=attn_heads, dim_head=attention_head_dim
                                ),
                            )
                        )
                        if use_attn
                        else nn.Identity(),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = self.channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_channels, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, time, x_self_cond=None, condition=None, class_condition=None):
        # Concatenate inputs internally
        if self.self_condition and x_self_cond is not None:
            x = torch.cat((x, x_self_cond), dim=1)

        if exists(condition):
            x = torch.cat((x, condition), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        if hasattr(self, "class_emb") and exists(class_condition):
            class_condition = self.class_emb(class_condition)
            class_condition = rearrange(class_condition, "b -> b 1 1 1")
            t = t + class_condition

        h = []

        for resnet1, resnet2, attn, downsample in self.downs:
            x = resnet1(x, t)
            h.append(x)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
