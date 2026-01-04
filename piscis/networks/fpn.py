import torch
import torch.nn as nn

from functools import partial
from typing import Any, Optional, Sequence, Tuple, Union

from piscis.networks.conv import Conv

ModuleDef = Any

BatchActConv = partial(Conv, bias=True, layers=['bn', 'act', 'conv'])
BatchConv = partial(Conv, bias=True, layers=['bn', 'conv'])


class BatchConvStyle(nn.Module):

    """Convolutional block with batch norm, activation, and style transfer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    style_channels : int
        Number of style channels.
    kernel_size : Union[int, Sequence[int]]
        Size of the convolutional kernel.
    conv : ModuleDef
        Convolution module.
    dense : ModuleDef
        Dense module.
    bn : ModuleDef
        Batch norm module.
    act : ModuleDef
        Activation function.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_channels: int,
            kernel_size: Union[int, Sequence[int]],
            conv: ModuleDef,
            dense: ModuleDef,
            bn: ModuleDef,
            act: ModuleDef
    ) -> None:
        
        super().__init__()

        self.conv = BatchActConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            conv=conv,
            bn=bn,
            act=act
        )

        self.dense = dense(
            in_features=style_channels,
            out_features=out_channels
        )

    def forward(
            self,
            style: Optional[torch.Tensor],
            x: torch.Tensor,
            y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if y is not None:
            x = x + y

        if style is not None:
            feat = self.dense(style)
            x = x + feat[:, :, None, None]

        x = self.conv(x)

        return x
    

class UpConv(nn.Module):

    """Upsampling convolutional block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    style_channels : int
        Number of style channels.
    kernel_size : Union[int, Sequence[int]]
        Size of the convolutional kernel.
    conv : ModuleDef
        Convolution module.
    dense : ModuleDef
        Dense module.
    bn : ModuleDef
        Batch norm module.
    act : ModuleDef
        Activation function.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            style_channels: int,
            kernel_size: Union[int, Sequence[int]],
            conv: ModuleDef,
            dense: ModuleDef,
            bn: ModuleDef,
            act: ModuleDef
    ) -> None:

        super().__init__()
        
        self.proj = BatchConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            conv=conv,
            bn=bn
        )

        self.conv = BatchActConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            conv=conv,
            bn=bn,
            act=act
        )

        convs = partial(
            BatchConvStyle,
            in_channels=out_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            kernel_size=kernel_size,
            conv=conv,
            dense=dense,
            bn=bn,
            act=act
        )

        self.convs_0 = convs()
        self.convs_1 = convs()
        self.convs_2 = convs()

    def forward(
            self,
            x: torch.Tensor,
            y: Optional[torch.Tensor],
            style: Optional[torch.Tensor]
    ) -> torch.Tensor:

        x = self.proj(x) + self.convs_0(style, self.conv(x), y)
        x = x + self.convs_2(style, self.convs_1(style, x))

        return x
    

class MakeStyle(nn.Module):

    """Style transfer module."""
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        style = x.mean(dim=tuple(range(2, x.ndim)))
        style = style / torch.linalg.norm(style, dim=-1, keepdim=True)

        return style


class Decoder(nn.Module):

    """Decoder module.

    Parameters
    ----------
    stage_sizes : Sequence[int]
        Number of channels at each stage.
    kernel_size : Union[int, Sequence[int]]
        Size of the convolutional kernel.
    conv : ModuleDef
        Convolution module.
    dense : ModuleDef
        Dense module.
    bn : ModuleDef
        Batch norm module.
    act : ModuleDef
        Activation function.
    """

    def __init__(
            self,
            stage_sizes: Sequence[int],
            kernel_size: Union[int, Sequence[int]],
            conv: ModuleDef,
            dense: ModuleDef,
            bn: ModuleDef,
            act: ModuleDef
    ) -> None:
        
        super().__init__()

        self.stage_sizes = stage_sizes

        up = partial(
            UpConv,
            style_channels=stage_sizes[-1],
            kernel_size=kernel_size,
            conv=conv,
            dense=dense,
            bn=bn,
            act=act
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.up_blocks = nn.ModuleList()
        self.up_blocks.append(
            up(
                in_channels=self.stage_sizes[-1],
                out_channels=self.stage_sizes[-1]
            )
        )
        for i in range(1, len(self.stage_sizes)):
            self.up_blocks.append(up(in_channels=self.stage_sizes[-i], out_channels=self.stage_sizes[-i - 1]))

        self.resize_up_blocks = nn.ModuleList()
        self.out_channels = self.stage_sizes[0]
        for i in range(len(self.stage_sizes) - 1):
            resize_up_block = nn.ModuleList()
            in_channels = self.stage_sizes[-1 - i]
            for j in range(len(self.stage_sizes) - 1 - i):
                if j > 0:
                    in_channels = self.out_channels
                resize_up_block.append(
                    up(in_channels=in_channels, out_channels=self.out_channels)
                )
            self.resize_up_blocks.append(resize_up_block)
        
    def forward(
            self,
            style: torch.Tensor,
            xd: Sequence[torch.Tensor]
    ) -> torch.Tensor:

        f = xd[-1]
        feature_maps = None

        for i, up in enumerate(self.up_blocks):
            f = up(f, xd[-i - 1], style)
            f_up = f
            if i < len(self.resize_up_blocks):
                for resize_up_block in self.resize_up_blocks[i]:
                    f_up = self.upsample(f_up)
                    f_up = resize_up_block(f_up, None, style)
            if feature_maps is None:
                feature_maps = f_up
            else:
                feature_maps = feature_maps + f_up
            f = self.upsample(f)

        return feature_maps


class FPN(nn.Module):

    """Feature pyramid network.

    Parameters
    ----------
    encoder : ModuleDef
        Encoder module.
    encoder_levels : Sequence[int]
        Encoder levels to use for the feature pyramid.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : Union[int, Sequence[int]], optional
        Size of the convolutional kernel. Default is s3.
    style : bool, optional
        Whether to use style transfer. Default is True.
    bn_momentum : float, optional
        Momentum parameter for batch norm layers. Default is 0.1.
    bn_epsilon : float, optional
        Epsilon parameter for batch norm layers. Default is 1e-5.
    conv : ModuleDef, optional
        Convolution module. Default is nn.Conv2d.
    dense : ModuleDef, optional
        Dense module. Default is nn.Linear.
    bn : ModuleDef, optional
        Batch norm module. Default is nn.BatchNorm2d.
    act : ModuleDef, optional
        Activation function. Default is nn.SiLU.
    """

    def __init__(
            self,
            encoder: ModuleDef,
            encoder_levels: Sequence[int],
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Sequence[int]] = 3,
            style: bool = True,
            bn_momentum: float = 0.1,
            bn_epsilon: float = 1e-05,
            conv: ModuleDef = nn.Conv2d,
            dense: ModuleDef = nn.Linear,
            bn: ModuleDef = nn.BatchNorm2d,
            act: ModuleDef = nn.SiLU
    ) -> None:
        
        super().__init__()

        self.encoder = encoder(in_channels=in_channels)
        self.encoder_levels = sorted(set(encoder_levels))
        self.style = style

        if self.style:
            self.make_style = MakeStyle()

        bn = partial(
            bn,
            eps=bn_epsilon,
            momentum=bn_momentum
        )

        self.decoder = Decoder(
            stage_sizes=[self.encoder.stage_sizes[i] for i in self.encoder_levels],
            kernel_size=kernel_size,
            conv=conv,
            dense=dense,
            bn=bn,
            act=act
        )

        self.output = BatchActConv(
            in_channels=self.decoder.out_channels,
            out_channels=out_channels,
            kernel_size=1,
            conv=conv,
            bn=bn,
            act=act
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # Get encoder outputs.
        x = self.encoder(x, capture_list=self.encoder_levels)
        x = [x[i] for i in self.encoder_levels]

        # Make style vectors if necessary.
        if self.style:
            style = self.make_style(x[-1])
        else:
            style = None

        # Apply decoder and output block.
        x = self.decoder(style, x)
        x = self.output(x)

        return x, style
