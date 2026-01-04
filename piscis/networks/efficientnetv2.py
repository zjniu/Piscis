import copy
import math
import torch
import torch.nn as nn

from functools import partial
from typing import Any, Dict, Optional, Sequence, Union

from piscis.networks.conv import ConvBatchAct, MBConv, FusedMBConv
from piscis.networks.efficientnetv2_defaults import DEFAULT_BLOCKS_ARGS

ModuleDef = Any


def round_features(
        features: int,
        width_coefficient: float,
        min_depth: int,
        depth_divisor: int
) -> int:

    """Round the number of features based on the depth multiplier.

    Parameters
    ----------
    features : int
        Number of features.
    width_coefficient : float
        Scaling coefficient for network width.
    min_depth : int
        Minimum number of filters.
    depth_divisor : int
        Unit of network width.

    Returns
    -------
    new_features : int
        Rounded number of features.
    """

    features *= width_coefficient
    minimum_depth = min_depth or depth_divisor
    new_features = max(
        minimum_depth,
        int(features + depth_divisor / 2) // depth_divisor * depth_divisor,
    )
    new_features = int(new_features)

    return new_features


def round_repeats(
        repeats: int,
        depth_coefficient: float
) -> int:

    """Round number of repeats based on depth multiplier.

    Parameters
    ----------
    repeats : int
        Number of repeats.
    depth_coefficient : float
        Scaling coefficient for network depth.

    Returns
    -------
    new_repeats : int
        Rounded number of repeats.
    """

    new_repeats = int(math.ceil(depth_coefficient * repeats))

    return new_repeats


class EfficientNetV2(nn.Module):

    """EfficientNetV2 architecture.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    stem_stride : int
        Stride of the stem convolution.
    blocks_args : Sequence
        List of arguments to construct block modules.
    stochastic_depth_prob : float
        Stochastic depth probability.
    bn_momentum : float
        Momentum parameter for batch norm layers.
    conv : ModuleDef
        Convolution module.
    bn : ModuleDef
        Batch norm module.
    act : Callable
        Activation function.
    """

    def __init__(
            self,
            in_channels: int,
            stem_stride: int,
            blocks_args: Sequence,
            stochastic_depth_prob: float,
            bn_momentum: float,
            conv: ModuleDef = nn.Conv2d,
            bn: ModuleDef = nn.BatchNorm2d,
            act: ModuleDef = nn.SiLU
    ) -> None:
        
        super().__init__()

        bn = partial(bn, momentum=bn_momentum)

        # Build stem.
        self.stem = ConvBatchAct(
            in_channels=in_channels,
            out_channels=blocks_args[0][0]['in_channels'],
            kernel_size=3,
            stride=stem_stride,
            bias=False,
            conv=conv,
            bn=bn,
            act=act
        )

        # Get pooling modules.
        stem_conv = self.stem.conv[0]
        if isinstance(stem_conv, nn.Conv2d):
            avg_pool = nn.AvgPool2d
            max_pool = nn.MaxPool2d
        elif isinstance(stem_conv, nn.Conv3d):
            avg_pool = nn.AvgPool3d
            max_pool = nn.MaxPool3d
        else:
            raise ValueError(f"Unsupported conv type {type(stem_conv)}.")
        
        # Build blocks.
        blocks_args = copy.deepcopy(blocks_args)
        b = 0
        n_blocks = float(sum(len(block_args) for block_args in blocks_args))

        self.blocks = nn.ModuleList()
        self.stage_sizes = []

        for block_args in blocks_args:

            block_layers = []

            for args in block_args:

                block = {0: MBConv, 1: FusedMBConv}[args.pop('conv_type')]

                pool = args.pop('pool')
                if pool == 'avg':
                    block_layers.append(avg_pool(kernel_size=2))
                elif pool == 'max':
                    block_layers.append(max_pool(kernel_size=2))

                block_layers.append(
                    block(
                        stochastic_depth_prob=stochastic_depth_prob * b / n_blocks,
                        conv=conv,
                        bn=bn,
                        act=act,
                        **args,
                    )
                )
                
                b += 1

            self.blocks.append(nn.Sequential(*block_layers))
            self.stage_sizes.append(args['out_channels'])

    def forward(
            self,
            x: torch.Tensor,
            capture_list: Optional[Sequence[int]] = None
    ) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        
        # Initialize capture list.
        captures = {}

        x = self.stem(x)

        for i, block in enumerate(self.blocks):

            x = block(x)

            # Capture intermediate output if necessary.
            if (capture_list is not None) and (i in capture_list):
                captures[i] = x

        if capture_list is None:
            output = x
        else:
            output = captures

        return output


def build_efficientnetv2(
        blocks_args: Sequence[Dict[str, Any]],
        width_coefficient: float,
        depth_coefficient: float,
        stem_stride: int = 2,
        stochastic_depth_prob: float = 0.2,
        bn_momentum: float = 0.1,
        depth_divisor: int = 8,
        min_depth: int = 8,
        conv: ModuleDef = nn.Conv2d,
        bn: ModuleDef = nn.BatchNorm2d,
        act: ModuleDef = nn.SiLU
) -> partial:

    """Build EfficientNetV2 architecture.

    Parameters
    ----------
    blocks_args : Sequence[Dict[str, Any]]
        List of dictionaries of arguments to construct block modules.
    width_coefficient : float
        Scaling coefficient for network width.
    depth_coefficient : float
        Scaling coefficient for network depth.
    stem_stride : int, optional
        Stride of the stem convolution. Default is 2.
    stochastic_depth_prob : float, optional
        Stochastic depth probability. Default is 0.2.
    bn_momentum : float, optional
        Momentum parameter for batch norm layers. Default is 0.1.
    depth_divisor : int, optional
        Unit of network width. Default is 8.
    min_depth : int, optional
        Minimum number of filters. Default is 8.
    conv : ModuleDef, optional
        Convolution module. Default is nn.Conv2d.
    bn : ModuleDef, optional
        Batch norm module. Default is nn.BatchNorm2d.
    act : ModuleDef, optional
        Activation function. Default is nn.SiLU.

    Returns
    -------
    model : partial
        EfficientNetV2 architecture.
    """

    blocks_args = copy.deepcopy(blocks_args)

    long_blocks_args = []
    b = 0
    for block_args in blocks_args:

        long_block_args = []

        assert block_args['num_repeat'] > 0

        # Update block input and output features based on depth multiplier.
        block_args['in_channels'] = round_features(
            features=block_args['in_channels'],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor
        )
        block_args['out_channels'] = round_features(
            features=block_args['out_channels'],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor
        )

        repeats = round_repeats(
            repeats=block_args.pop('num_repeat'),
            depth_coefficient=depth_coefficient
        )
        for j in range(repeats):

            # The first block needs to take care of stride and filter size increases.
            if j > 0:
                block_args['pool'] = None
                block_args['stride'] = 1
                block_args['in_channels'] = block_args['out_channels']

            long_block_args.append(block_args.copy())

            b += 1

        long_blocks_args.append(long_block_args)

    model = partial(
        EfficientNetV2,
        stem_stride=stem_stride,
        blocks_args=long_blocks_args,
        stochastic_depth_prob=stochastic_depth_prob,
        bn_momentum=bn_momentum,
        conv=conv,
        bn=bn,
        act=act
    )

    return model


EfficientNetV2B0 = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-b0'],
    width_coefficient=1.0,
    depth_coefficient=1.0,
)

EfficientNetV2B1 = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-b1'],
    width_coefficient=1.0,
    depth_coefficient=1.1,
)

EfficientNetV2B2 = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-b2'],
    width_coefficient=1.1,
    depth_coefficient=1.2,
)

EfficientNetV2B3 = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-b3'],
    width_coefficient=1.2,
    depth_coefficient=1.4,
)

EfficientNetV2S = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-s'],
    width_coefficient=1.0,
    depth_coefficient=1.0,
)

EfficientNetV2M = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-m'],
    width_coefficient=1.0,
    depth_coefficient=1.0,
)

EfficientNetV2L = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-l'],
    width_coefficient=1.0,
    depth_coefficient=1.0,
)
