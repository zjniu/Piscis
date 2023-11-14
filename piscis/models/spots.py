import jax
import numpy as np

from flax import linen as nn
from typing import Tuple, Union

from piscis.networks.efficientnetv2 import build_efficientnetv2
from piscis.networks.fpn import FPN

BLOCK_ARGS = [
    {
        "kernel_size": (3, 3),
        "num_repeat": 4,
        "features_in": 32,
        "features_out": 32,
        "expand_ratio": 1,
        "se_ratio": 0.0,
        "strides": 1,
        "conv_type": 1,
        "pool": None
    }, {
        "kernel_size": (3, 3),
        "num_repeat": 4,
        "features_in": 32,
        "features_out": 64,
        "expand_ratio": 2,
        "se_ratio": 0.0,
        "strides": 1,
        "conv_type": 1,
        "pool": "max"
    }, {
        "conv_type": 0,
        "expand_ratio": 4,
        "features_in": 64,
        "kernel_size": (3, 3),
        "num_repeat": 4,
        "features_out": 128,
        "se_ratio": 0.25,
        "strides": 1,
        "pool": "max"
    }, {
        "conv_type": 0,
        "expand_ratio": 4,
        "features_in": 128,
        "kernel_size": (3, 3),
        "num_repeat": 4,
        "features_out": 256,
        "se_ratio": 0.25,
        "strides": 1,
        "pool": "max"
    }
]


class SpotsModel(nn.Module):

    """Spot detection model.

    Attributes
    ----------
    style : bool
        Whether to use style transfer.
    aggregate : str
        Aggregation mode for the feature pyramid network. Supported modes are 'sum' and 'concatenate'.
    dropout_rate : float
        Dropout rate at skip connections.
    """

    style: bool = True
    aggregate: str = 'sum'
    dropout_rate: float = 0.2

    @nn.compact
    def __call__(
            self,
            x: jax.Array,
            train: bool = True,
            return_style: bool = False
    ) -> Union[Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]:

        encoder = build_efficientnetv2(
            blocks_args=BLOCK_ARGS,
            model_name='EfficientNetV2XS',
            width_coefficient=1.0,
            depth_coefficient=1.0,
            stem_strides=1,
            dropout_rate=self.dropout_rate
        )

        x, style = FPN(
            encoder=encoder,
            encoder_levels={0, 1, 2, 3},
            features=3,
            style=self.style,
            aggregate=self.aggregate,
            dropout_rate=self.dropout_rate
        )(x, train=train)
        deltas = x[:, :, :, :2]
        labels = nn.sigmoid(x[:, :, :, 2:3])

        if return_style:
            return deltas, labels, style
        else:
            return deltas, labels


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

    stride_scale = np.prod([block['strides'] for block in BLOCK_ARGS])
    pool_scale = 2 ** sum((0 if block['pool'] is None else 1 for block in BLOCK_ARGS if block['pool']))
    scale = stride_scale * pool_scale
    rounded_input_size = scale * np.rint(np.array(input_size) / scale).astype(int)
    rounded_input_size = (int(rounded_input_size[0]), int(rounded_input_size[1]))

    return rounded_input_size
