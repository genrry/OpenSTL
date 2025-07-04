import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


def exists(x):
    return x is not None


RELU_LEAK = 0.2


class UNetBlock(torch.nn.Module):
    def __init__(
        self,
        in_chans,
        dim_out,
        time_emb_dim=None,
        transposed=False,
        bn=True,
        relu=True,
        size=4,
        pad=1,
        dropout=0.0,
    ):
        super().__init__()
        batch_norm = bn
        relu_leak = None if relu else RELU_LEAK
        kern_size = size
        self.time_mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        ops = []
        if not transposed:
            ops.append(
                torch.nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=dim_out,
                    kernel_size=kern_size,
                    stride=2,
                    padding=pad,
                    bias=True,
                )
            )
        else:
            ops.append(torch.nn.Upsample(scale_factor=2, mode="bilinear"))
            ops.append(
                torch.nn.Conv2d(
                    in_channels=in_chans,
                    out_channels=dim_out,
                    kernel_size=(kern_size - 1),
                    stride=1,
                    padding=pad,
                    bias=True,
                )
            )
        if batch_norm:
            ops.append(torch.nn.BatchNorm2d(dim_out))
        else:
            ops.append(nn.GroupNorm(8, dim_out))

        self.ops = torch.nn.Sequential(*ops)

        if relu_leak is None or relu_leak == 0:
            self.act = torch.nn.ReLU()
        else:
            self.act = torch.nn.LeakyReLU(negative_slope=relu_leak)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb=None):
        x = self.ops(x)
        if exists(self.time_mlp):
            assert exists(time_emb), "Time embedding must be provided if time_mlp is not None"
            time_emb = self.time_mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        x = self.dropout(x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        channels: int = 3,
        out_channels: int = None,
        condition_channels: int = 0,
        with_time_emb: bool = False,
        outer_sample_mode: str = "bilinear",  # bilinear or nearest
        upsample_dims: tuple = None,
        dropout: float = 0.0,
        input_dropout: float = 0.0,
        dim_mults: tuple = (1, 2, 4, 8),
        **kwargs,
    ):
        super().__init__()
        self.outer_sample_mode = outer_sample_mode
        if upsample_dims is None:
            self.upsampler = nn.Identity()
        else:
            self.upsampler = torch.nn.Upsample(
                size=tuple(upsample_dims), mode=self.outer_sample_mode
            )

        self.channels = channels
        self.out_channels = out_channels if out_channels is not None else channels
        self.condition_channels = condition_channels
        in_channels = self.channels + self.condition_channels

        # Build network operations
        if with_time_emb:
            # time embeddings
            self.time_dim = dim * 2
            self.time_emb_mlp = nn.Sequential(
                nn.Linear(self.time_dim, self.time_dim),
                nn.GELU(),
                nn.Linear(self.time_dim, self.time_dim),
            )
        else:
            self.time_dim = None
            self.time_emb_mlp = None

        self.init_conv = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.dropout_input = nn.Dropout(input_dropout)

        block_kwargs = dict(time_emb_dim=self.time_dim, dropout=dropout)

        # encoder layers
        self.input_ops = torch.nn.ModuleList()
        dims = [dim] + [dim * m for m in dim_mults]
        in_out = list(zip(dims[:-1], dims[1:]))

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (len(in_out) - 1)
            self.input_ops.append(
                UNetBlock(
                    dim_in, dim_out, transposed=False, bn=not is_last, relu=False, **block_kwargs
                )
            )

        # decoder layers
        self.output_ops = torch.nn.ModuleList()
        reversed_dims = list(reversed(dims))
        in_out_rev = list(zip(reversed_dims[:-1], reversed_dims[1:]))

        in_ch = reversed_dims[0]
        for i, (_, dim_out) in enumerate(in_out_rev):
            self.output_ops.append(
                UNetBlock(in_ch, dim_out, transposed=True, bn=True, relu=True, **block_kwargs)
            )
            in_ch = dim_out * 2

        self.readout = torch.nn.Sequential(
            # torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(
                in_channels=dim,
                out_channels=self.out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=True,
            ),
        )

        # Initialize weights
        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(module):
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            module.weight.data.normal_(0.0, 0.02)
        elif isinstance(module, torch.nn.BatchNorm2d):
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)

    def _apply_ops(self, x: Tensor, time: Tensor = None):
        skip_connections = []
        # Encoder ops
        x = self.init_conv(x)
        x = self.dropout_input(x)
        for op in self.input_ops:
            x = op(x, time)
            skip_connections.append(x)
        # Decoder ops
        x = skip_connections.pop()
        for op in self.output_ops:
            x = op(x, time)
            if skip_connections:
                x = torch.cat([x, skip_connections.pop()], dim=1)
        x = self.readout(x)
        return x

    def forward(self, inputs, time=None, condition=None, return_time_emb: bool = False, **kwargs):
        # Preprocess inputs for shape
        if self.condition_channels > 0:
            x = torch.cat([inputs, condition], dim=1)
        else:
            x = inputs
            assert condition is None

        t = self.time_emb_mlp(time) if exists(self.time_emb_mlp) and exists(time) else None

        # Apply operations
        orig_x_shape = x.shape[-2:]
        x = self.upsampler(x)
        y = self._apply_ops(x, t)
        y = torch.nn.functional.interpolate(y, size=orig_x_shape, mode=self.outer_sample_mode)

        return y
