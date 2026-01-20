import torch
import torch.nn as nn

from functools import partial
from torchvision.ops import StochasticDepth
from torchvision.utils import _make_ntuple
from typing import Any, Optional, Sequence, Tuple, Union

ModuleDef = Any

class Conv(nn.Module):

    """Convolutional block with batch norm and activation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : Union[int, Sequence[int]]
        Size of the convolutional kernel. Default is 3.
    stride : Union[int, Sequence[int]]
        Stride of the convolution. Default is 1.
    padding : Optional[Union[int, Sequence[int], str]], optional
        Padding of the convolution. Default is None.
    dilation : Union[int, Sequence[int]], optional
        Dilation of the convolution. Default is 1.
    groups : int, optional
        Number of groups of the convolution. Default is 1.
    bias : Optional[bool], optional
        Whether to use bias in the convolution. Default is None.
    padding_mode : str, optional
        Padding mode of the convolution. Default is 'zeros'.
    conv : ModuleDef, optional
        Convolution module. Default is nn.Conv2d.
    bn : ModuleDef, optional
        Batch norm module. Default is nn.BatchNorm2d.
    act : ModuleDef, optional
        Activation function. Default is nn.SiLU.
    layers : Sequence[str], optional
        Sequence of layers to apply. Default is ('conv', 'bn', 'act').
    """
    
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Sequence[int]] = 3,
            stride: Union[int, Sequence[int]] = 1,
            padding: Optional[Union[int, Sequence[int], str]] = None,
            dilation: Union[int, Sequence[int]] = 1,
            groups: int = 1,
            bias: Optional[bool] = None,
            padding_mode: str = 'zeros',
            conv: ModuleDef = nn.Conv2d,
            bn: ModuleDef = nn.BatchNorm2d,
            act: ModuleDef = nn.SiLU,
            layers: Sequence[str] = ('conv', 'bn', 'act')
    ) -> None:
        
        super().__init__()

        layer_list = []
        channels = in_channels

        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))

        for layer in layers:
            if layer == 'conv':
                layer_list.append(
                    conv(
                        in_channels=channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias,
                        padding_mode=padding_mode
                    )
                )
                channels = out_channels
            elif layer == 'bn':
                layer_list.append(bn(num_features=channels)
                )
            elif layer == 'act':
                layer_list.append(act())
            else:
                raise ValueError(f"Unknown layer {layer}.")

        self.conv = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.conv(x)
    

ConvBatchAct = partial(Conv, bias=False, layers=('conv', 'bn', 'act'))


class SqueezeExcite(nn.Module):

    """Squeeze and excite block.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    squeeze_channels : int
        Number of squeeze channels.
    conv : ModuleDef, optional
        Convolution module. Default is nn.Conv2d.
    act : ModuleDef, optional
        Activation function. Default is nn.SiLU.
    """

    def __init__(
            self,
            in_channels: int,
            squeeze_channels: int,
            conv: ModuleDef = nn.Conv2d,
            act: ModuleDef = nn.SiLU
    ) -> None:
        
        super().__init__()

        conv = partial(conv, kernel_size=1, stride=1, bias=True)
        self.reduce = conv(in_channels, squeeze_channels)
        self.act = act()
        self.expand = conv(squeeze_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

        if isinstance(self.reduce, nn.Conv2d):
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        elif isinstance(self.reduce, nn.Conv3d):
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
        else:
            raise ValueError(f"Unsupported conv type {type(self.reduce)}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        se = self.avg_pool(x)
        se = self.reduce(se)
        se = self.act(se)
        se = self.expand(se)
        se = self.sigmoid(se)
        x = x * se

        return x


class MBConv(nn.Module):

    """Mobile inverted bottleneck convolutional block.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    expand_ratio : int
        Expansion ratio.
    kernel_size : Union[int, Sequence[int]]
        Size of the convolutional kernel.
    stride : int
        Stride of the convolution.
    se_ratio : float
        Squeeze and excitation ratio.
    stochastic_depth_prob : float
        Stochastic depth probability.
    conv : ModuleDef, optional
        Convolution module. Default is nn.Conv2d.
    bn : ModuleDef, optional
        Batch norm module. Default is nn.BatchNorm2d.
    act : ModuleDef, optional
        Activation function. Default is nn.SiLU.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: int,
            kernel_size: Union[int, Sequence[int]],
            stride: int,
            se_ratio: float,
            stochastic_depth_prob: float,
            conv: ModuleDef = nn.Conv2d,
            bn: ModuleDef = nn.BatchNorm2d,
            act: ModuleDef = nn.SiLU
    ) -> None:
        
        super().__init__()

        block_layers = []

        # Expansion phase.
        filters = in_channels * expand_ratio
        if expand_ratio != 1:
            block_layers.append(
                ConvBatchAct(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=1,
                    stride=1,
                    conv=conv,
                    bn=bn,
                    act=act
                )
            )
        
        # Depthwise convolution.
        block_layers.append(
            ConvBatchAct(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=stride,
                groups=filters,
                conv=conv,
                bn=bn,
                act=act
            )
        )

        # Squeeze and excite.
        if 0 < se_ratio <= 1:
            squeeze_channels = max(1, int(in_channels * se_ratio))
            block_layers.append(
                SqueezeExcite(
                    in_channels=filters,
                    squeeze_channels=squeeze_channels,
                    conv=conv,
                    act=act
                )
            )

        # Output phase.
        block_layers.append(
            Conv(
                in_channels=filters,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                conv=conv,
                bn=bn,
                act=None,
                layers=['conv', 'bn']
            )
        )

        self.block = nn.Sequential(*block_layers)

        # Residual.
        self.use_res_connect = (stride == 1) and (in_channels == out_channels)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, mode='row')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input = x

        x = self.block(x)

        if self.use_res_connect:
            x = self.stochastic_depth(x)
            x = x + input

        return x


class FusedMBConv(nn.Module):

    """Fused mobile inverted bottleneck convolutional block.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    expand_ratio : int
        Expansion ratio.
    kernel_size : Union[int, Sequence[int]]
        Size of the convolutional kernel.
    stride : int
        Stride of the convolution.
    se_ratio : float
        Squeeze and excitation ratio.
    stochastic_depth_prob : float
        Stochastic depth probability.
    conv : ModuleDef, optional
        Convolution module. Default is nn.Conv2d.
    bn : ModuleDef, optional
        Batch norm module. Default is nn.BatchNorm2d.
    act : ModuleDef, optional
        Activation function. Default is nn.SiLU.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expand_ratio: int,
            kernel_size: Union[int, Sequence[int]],
            stride: int,
            se_ratio: float,
            stochastic_depth_prob: float,
            conv: ModuleDef = nn.Conv2d,
            bn: ModuleDef = nn.BatchNorm2d,
            act: ModuleDef = nn.SiLU
    ) -> None:
        
        super().__init__()

        block_layers = []

        # Expansion phase.
        filters = in_channels * expand_ratio
        if expand_ratio != 1:
            block_layers.append(
                ConvBatchAct(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    conv=conv,
                    bn=bn,
                    act=act
                )
            )

        # Squeeze and excite.
        if 0 < se_ratio <= 1:
            squeeze_channels = max(1, int(in_channels * se_ratio))
            block_layers.append(
                SqueezeExcite(
                    in_channels=filters,
                    squeeze_channels=squeeze_channels,
                    conv=conv,
                    act=act
                )
            )

        # Output phase.
        if expand_ratio == 1:
            block_layers.append(
                ConvBatchAct(
                    in_channels=filters,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    conv=conv,
                    bn=bn,
                    act=act
                )
            )
        else:
            block_layers.append(
                Conv(
                    in_channels=filters,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    conv=conv,
                    bn=bn,
                    act=None,
                    layers=['conv', 'bn']
                )
            )

        self.block = nn.Sequential(*block_layers)

        # Residual.
        self.use_res_connect = (stride == 1) and (in_channels == out_channels)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, mode='row')

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input = x

        x = self.block(x)

        if self.use_res_connect:
            x = self.stochastic_depth(x)
            x = x + input

        return x
