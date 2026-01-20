import numpy as np
import torch
import torch.nn as nn

from functools import partial
from typing import Sequence, Tuple, Union

from piscis.networks.efficientnetv2 import build_efficientnetv2
from piscis.networks.fpn import FPN
from piscis.utils import vmap_deformable_max_pool, vmap_deformable_sum_pool

BLOCK_ARGS = [
    {
        "kernel_size": (3, 3),
        "num_repeat": 4,
        "in_channels": 32,
        "out_channels": 32,
        "expand_ratio": 1,
        "se_ratio": 0.0,
        "stride": 1,
        "conv_type": 1,
        "pool": None
    }, {
        "kernel_size": (3, 3),
        "num_repeat": 4,
        "in_channels": 32,
        "out_channels": 64,
        "expand_ratio": 2,
        "se_ratio": 0.0,
        "stride": 1,
        "conv_type": 1,
        "pool": "max"
    }, {
        "conv_type": 0,
        "expand_ratio": 4,
        "in_channels": 64,
        "kernel_size": (3, 3),
        "num_repeat": 4,
        "out_channels": 128,
        "se_ratio": 0.25,
        "stride": 1,
        "pool": "max"
    }, {
        "conv_type": 0,
        "expand_ratio": 4,
        "in_channels": 128,
        "kernel_size": (3, 3),
        "num_repeat": 4,
        "out_channels": 256,
        "se_ratio": 0.25,
        "stride": 1,
        "pool": "max"
    }
]


class SpotsModel(nn.Module):

    """Spot detection model.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels. Default is 1.
    style : bool, optional
        Whether to use style transfer. Default is True.
    pooling : str, optional
        Pooling type applied to labels. Supported types are 'max' and 'sum'. Default is 'max'.
    stochastic_depth_prob : float, optional
        Stochastic depth probability. Default is 0.0.
    kernel_size : Sequence[int], optional
        Kernel size or window size of the max pooling operation. Default is (3, 3).
    """

    def __init__(
            self,
            in_channels: int = 1,
            stochastic_depth_prob: float = 0.0,
            style: bool = True,
            pooling: str = 'max',
            kernel_size: Sequence[int] = (3, 3)
    ) -> None:

        super().__init__()

        act = partial(nn.SiLU, inplace=True)

        encoder = build_efficientnetv2(
            blocks_args=BLOCK_ARGS,
            width_coefficient=1.0,
            depth_coefficient=1.0,
            stem_stride=1,
            stochastic_depth_prob=stochastic_depth_prob,
            conv=Conv2d,
            act=act
        )

        self.fpn = FPN(
            encoder=encoder,
            encoder_levels=(0, 1, 2, 3),
            in_channels=in_channels,
            out_channels=3,
            style=style,
            conv=Conv2d,
            dense=Linear,
            act=act
        )

        self.sigmoid = nn.Sigmoid()
        if pooling == 'max':
            self.pool = vmap_deformable_max_pool
        elif pooling == 'sum':
            self.pool = vmap_deformable_sum_pool
        else:
            raise ValueError(f"Pooling type is not supported.")
        self.kernel_size = kernel_size

    def forward(
            self,
            x: torch.Tensor,
            return_style: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        x, style = self.fpn(x)
        labels = self.sigmoid(x[:, 0])
        deltas = x[:, 1:]
        labels = self.pool(labels, deltas, self.kernel_size)

        if return_style:
            return labels, deltas, style
        else:
            return labels, deltas


class Conv2d(nn.Conv2d):
    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class Linear(nn.Linear):

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def round_input_size(input_size: Tuple[int, int]) -> Tuple[int, int]:

    """Round SpotsModel input size.

    Parameters
    ----------
    input_size : Tuple[int, int]
        Input size.

    Returns
    -------
    rounded_input_size : Tuple[int, int]
        Rounded input size.
    """

    stride_scale = np.prod([block['stride'] for block in BLOCK_ARGS])
    pool_scale = 2 ** sum((0 if block['pool'] is None else 1 for block in BLOCK_ARGS if block['pool']))
    scale = stride_scale * pool_scale
    rounded_input_size = scale * np.ceil(np.array(input_size) / scale).astype(int)
    rounded_input_size = (int(rounded_input_size[0]), int(rounded_input_size[1]))

    return rounded_input_size
