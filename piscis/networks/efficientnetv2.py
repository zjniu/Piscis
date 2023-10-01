import copy
import jax
import math

from flax import linen as nn
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Union

from piscis.networks.conv import MBConv, FusedMBConv
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

    Attributes
    ----------
    stem_strides : int
        Strides of the stem convolution.
    blocks_args : Sequence
        List of arguments to construct block modules.
    dropout_rate : float
        Dropout rate at skip connections.
    bn_momentum : float
        Momentum parameter for batch norm layers.
    conv : ModuleDef
        Convolution module.
    dropout : ModuleDef
        Dropout module.
    bn : ModuleDef
        Batch norm module.
    act : Callable
        Activation function.
    """

    stem_strides: int
    blocks_args: Sequence
    dropout_rate: float
    bn_momentum: float
    conv: ModuleDef = nn.Conv
    dropout: ModuleDef = nn.Dropout
    bn: ModuleDef = nn.BatchNorm
    act: Callable = nn.swish

    @nn.compact
    def __call__(
            self,
            x: jax.Array,
            train: bool = True,
            capture_list: Optional[Sequence[int]] = None
    ) -> Union[jax.Array, Dict[int, jax.Array]]:

        # Initialize capture list.
        captures = {}

        bn = partial(
            self.bn,
            use_running_average=not train,
            momentum=self.bn_momentum
        )

        # Build stem.
        x = self.conv(
            features=self.blocks_args[0][0]['features_in'],
            kernel_size=(3, 3),
            strides=self.stem_strides,
            padding='SAME',
            use_bias=False,
            name='stem_conv',
        )(x)
        x = bn(
            name='stem_bn'
        )(x)
        x = self.act(x)

        # Build blocks.
        blocks_args = copy.deepcopy(self.blocks_args)
        b = 0
        blocks = float(sum(len(block_args) for block_args in blocks_args))

        for i, block_args in enumerate(blocks_args):

            for args in block_args:

                args = args.unfreeze()
                block = {0: MBConv, 1: FusedMBConv}[args.pop('conv_type')]

                if self.dropout_rate > 0:
                    dropout = partial(
                        self.dropout,
                        rate=self.dropout_rate * b / blocks,
                        deterministic=not train
                    )
                else:
                    dropout = None

                pool = args.pop('pool')
                if pool == 'avg':
                    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                elif pool == 'max':
                    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

                x = block(
                    conv=self.conv,
                    dropout=dropout,
                    bn=bn,
                    act=self.act,
                    **args,
                )(x)
                b += 1

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
        model_name: str,
        width_coefficient: float,
        depth_coefficient: float,
        stem_strides: int = 2,
        dropout_rate: float = 0.2,
        bn_momentum: float = 0.9,
        depth_divisor: int = 8,
        min_depth: int = 8,
        conv: ModuleDef = nn.Conv,
        dropout: ModuleDef = nn.Dropout,
        bn: ModuleDef = nn.BatchNorm,
        act: Callable = nn.swish
) -> partial:

    """Build EfficientNetV2 architecture.

    Parameters
    ----------
    blocks_args : Sequence[Dict[str, Any]]
        List of dictionaries of arguments to construct block modules.
    model_name : str
        Model name.
    width_coefficient : float
        Scaling coefficient for network width.
    depth_coefficient : float
        Scaling coefficient for network depth.
    stem_strides : int, optional
        Strides of the stem convolution. Default is 2.
    dropout_rate : float, optional
        Dropout rate at skip connections. Default is 0.2.
    bn_momentum : float, optional
        Momentum parameter for batch norm layers. Default is 0.9.
    depth_divisor : int, optional
        Unit of network width. Default is 8.
    min_depth : int, optional
        Minimum number of filters. Default is 8.
    conv : ModuleDef, optional
        Convolution module. Default is nn.Conv.
    dropout : ModuleDef, optional
        Dropout module. Default is nn.Dropout.
    bn : ModuleDef, optional
        Batch norm module. Default is nn.BatchNorm.
    act : Callable, optional
        Activation function. Default is nn.swish.

    Returns
    -------
    model : partial
        EfficientNetV2 architecture.
    """

    blocks_args = copy.deepcopy(blocks_args)

    long_blocks_args = []
    b = 0
    for i, block_args in enumerate(blocks_args):

        long_block_args = []

        assert block_args['num_repeat'] > 0

        # Update block input and output features based on depth multiplier.
        block_args['features_in'] = round_features(
            features=block_args['features_in'],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor
        )
        block_args['features_out'] = round_features(
            features=block_args['features_out'],
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
                block_args['strides'] = 1
                block_args['features_in'] = block_args['features_out']

            long_block_args.append(block_args.copy())

            b += 1

        long_blocks_args.append(long_block_args)

    model = partial(
        EfficientNetV2,
        stem_strides=stem_strides,
        blocks_args=long_blocks_args,
        dropout_rate=dropout_rate,
        bn_momentum=bn_momentum,
        conv=conv,
        dropout=dropout,
        bn=bn,
        act=act,
        name=model_name
    )

    return model


EfficientNetV2B0 = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-b0'],
    model_name='efficientnetv2-b0',
    width_coefficient=1.0,
    depth_coefficient=1.0,
)

EfficientNetV2B1 = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-b1'],
    model_name='efficientnetv2-b1',
    width_coefficient=1.0,
    depth_coefficient=1.1,
)

EfficientNetV2B2 = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-b2'],
    model_name='efficientnetv2-b2',
    width_coefficient=1.1,
    depth_coefficient=1.2,
)

EfficientNetV2B3 = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-b3'],
    model_name='efficientnetv2-b3',
    width_coefficient=1.2,
    depth_coefficient=1.4,
)

EfficientNetV2S = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-s'],
    model_name='efficientnetv2-s',
    width_coefficient=1.0,
    depth_coefficient=1.0,
)

EfficientNetV2M = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-m'],
    model_name='efficientnetv2-m',
    width_coefficient=1.0,
    depth_coefficient=1.0,
)

EfficientNetV2L = build_efficientnetv2(
    blocks_args=DEFAULT_BLOCKS_ARGS['efficientnetv2-l'],
    model_name='efficientnetv2-l',
    width_coefficient=1.0,
    depth_coefficient=1.0,
)
