import jax
import jax.numpy as jnp

from flax import linen as nn
from functools import partial
from jax.image import resize
from typing import Any, Callable, Optional, Sequence, Tuple

from piscis.networks.conv import Conv

ModuleDef = Any

BatchActConv = partial(Conv, layers=['bn', 'act', 'conv'])
BatchConv = partial(Conv, layers=['bn', 'conv'], act=None)


class BatchConvStyle(nn.Module):

    """Convolutional block with batch norm, activation, and style transfer.

    Attributes
    ----------
    features : int
        Number of output features.
    kernel_size : Sequence[int]
        Size of the convolutional kernel.
    conv : ModuleDef
        Convolution module.
    dense : ModuleDef
        Dense module.
    dropout : ModuleDef
        Dropout module.
    bn : ModuleDef
        Batch norm module.
    act : Callable
        Activation function.
    """

    features: int
    kernel_size: Sequence[int]
    conv: ModuleDef
    dense: ModuleDef
    dropout: ModuleDef
    bn: ModuleDef
    act: Callable

    @nn.compact
    def __call__(
            self,
            style: Optional[jax.Array],
            x: jax.Array,
            y: Optional[jax.Array] = None
    ) -> jax.Array:

        conv = partial(
            BatchActConv,
            features=self.features,
            kernel_size=self.kernel_size,
            conv=self.conv,
            bn=self.bn,
            act=self.act
        )

        if y is not None:
            x = self.dropout()(x)
            x = x + y

        if style is not None:
            full = partial(
                self.dense,
                features=self.features
            )
            feat = full()(style)
            x = x + feat[:, None, None]

        x = conv()(x)

        return x


class UpConv(nn.Module):

    """Upsampling convolutional block.

    Attributes
    ----------
    features : int
        Number of output features.
    kernel_size : Sequence[int]
        Size of the convolutional kernel.
    conv : ModuleDef
        Convolution module.
    dense : ModuleDef
        Dense module.
    dropout : ModuleDef
        Dropout module.
    bn : ModuleDef
        Batch norm module.
    act : Callable
        Activation function.
    """

    features: int
    kernel_size: Sequence[int]
    conv: ModuleDef
    dense: ModuleDef
    dropout: ModuleDef
    bn: ModuleDef
    act: Callable

    @nn.compact
    def __call__(
            self,
            x: jax.Array,
            y: Optional[jax.Array],
            style: Optional[jax.Array]
    ) -> jax.Array:

        conv = partial(
            BatchActConv,
            features=self.features,
            kernel_size=self.kernel_size,
            conv=self.conv,
            bn=self.bn,
            act=self.act
        )

        convs = partial(
            BatchConvStyle,
            features=self.features,
            kernel_size=self.kernel_size,
            conv=self.conv,
            dense=self.dense,
            dropout=self.dropout,
            bn=self.bn,
            act=self.act
        )

        proj = partial(
            Conv,
            features=self.features,
            kernel_size=(1, 1),
            conv=self.conv,
            bn=self.bn,
            act=None,
            layers=['bn', 'conv']
        )

        x = proj()(x) + convs()(style, conv()(x), y)
        x = x + convs()(style, convs()(style, x))

        return x


class MakeStyle(nn.Module):

    """Style transfer module."""

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:

        style = jnp.mean(x, axis=(1, 2))
        style = style / jnp.linalg.norm(style, axis=1, keepdims=True)

        return style


class Decoder(nn.Module):

    """Decoder module.

    Attributes
    ----------
    kernel_size : Sequence[int]
        Size of the convolutional kernel.
    aggregate : str
        Aggregation mode for feature maps. Supported modes are 'sum' and 'concatenate'.
    conv : ModuleDef
        Convolution module.
    dense : ModuleDef
        Dense module.
    dropout : ModuleDef
        Dropout module.
    bn : ModuleDef
        Batch norm module.
    act : Callable
        Activation function.
    """

    kernel_size: Sequence[int]
    aggregate: str
    conv: ModuleDef
    dense: ModuleDef
    dropout: ModuleDef
    bn: ModuleDef
    act: Callable

    @nn.compact
    def __call__(
            self,
            style: jax.Array,
            xd: Sequence[jax.Array]
    ) -> jax.Array:

        up = partial(
            UpConv,
            kernel_size=self.kernel_size,
            conv=self.conv,
            dense=self.dense,
            dropout=self.dropout,
            bn=self.bn,
            act=self.act
        )

        stage_sizes = [x.shape[-1] for x in xd]

        # Create a list to store feature maps.
        feature_maps = []

        f = up(
            features=stage_sizes[-1],
        )(xd[-1], xd[-1], style)
        feature_maps.append(f)

        for i, features in reversed(list(enumerate(stage_sizes[:-1]))):
            f = _interpolate(f, scale=2, method='nearest')
            f = up(
                features=features,
            )(f, xd[i], style)
            feature_maps.append(f)

        # Resize feature maps.
        for i in range(len(feature_maps[:-1])):
            for j in range(len(feature_maps) - i - 1):
                feature_maps[i] = _interpolate(feature_maps[i], scale=2, method='nearest')
                feature_maps[i] = up(
                    features=stage_sizes[0]
                )(feature_maps[i], None, style)

        # Aggregate feature maps.
        if self.aggregate == 'sum':
            aggregate_feature_maps = jnp.sum(jnp.array(feature_maps), axis=0)
        elif self.aggregate == 'concatenate':
            aggregate_feature_maps = jnp.concatenate(feature_maps, axis=-1)
        else:
            raise ValueError(f"Aggregation mode is not supported.")

        return aggregate_feature_maps


class FPN(nn.Module):

    """Feature pyramid network.

    Attributes
    ----------
    encoder : ModuleDef
        Encoder module.
    encoder_levels : Sequence[int]
        Encoder levels to use for the feature pyramid.
    features : int
        Number of output features.
    kernel_size : Sequence[int]
        Size of the convolutional kernel.
    style : bool
        Whether to use style transfer.
    aggregate : str
        Aggregation mode for feature maps. Supported modes are 'sum' and 'concatenate'.
    dropout_rate : float
        Dropout rate at skip connections.
    bn_momentum : float
        Momentum parameter for batch norm layers.
    bn_epsilon : float
        Epsilon parameter for batch norm layers.
    conv : ModuleDef
        Convolution module.
    dense : ModuleDef
        Dense module.
    bn : ModuleDef
        Batch norm module.
    act : Callable
        Activation function.
    """

    encoder: ModuleDef
    encoder_levels: Sequence[int]
    features: int
    kernel_size: Sequence[int] = (3, 3)
    style: bool = True
    aggregate: str = 'sum'
    dropout_rate: float = 0.2
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-05
    conv: ModuleDef = nn.Conv
    dense: ModuleDef = nn.Dense
    dropout: ModuleDef = nn.Dropout
    bn: ModuleDef = nn.BatchNorm
    act: Callable = nn.swish

    @nn.compact
    def __call__(
            self,
            x: jax.Array,
            train: bool = True
    ) -> Tuple[jax.Array, Optional[jax.Array]]:

        dropout = partial(
            self.dropout,
            rate=self.dropout_rate,
            deterministic=not train
        )

        bn = partial(
            self.bn,
            use_running_average=not train,
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
        )

        decoder = partial(
            Decoder,
            kernel_size=self.kernel_size,
            aggregate=self.aggregate,
            conv=self.conv,
            dense=self.dense,
            dropout=dropout,
            bn=bn,
            act=self.act
        )

        output = partial(
            BatchActConv,
            features=self.features,
            kernel_size=(1, 1),
            conv=self.conv,
            bn=bn,
            act=self.act
        )

        # Get encoder outputs.
        x = self.encoder()(x, train=train, capture_list=self.encoder_levels)
        x = [x[i] for i in self.encoder_levels]

        # Make style vectors if necessary.
        if self.style:
            style = MakeStyle()(x[-1])
        else:
            style = None

        # Apply decoder and output block.
        x = decoder()(style, x)
        x = output()(x)

        return x, style


def _interpolate(
        x: jax.Array,
        scale: float,
        *args: Any,
        **kwargs: Any
):

    """Interpolate feature maps.

    Parameters
    ----------
    x : jax.Array
        Feature maps.
    scale : float
        Scale factor.
    *args : Any
        Positional arguments for the resize function.
    **kwargs : Any
        Keyword arguments for the resize function.

    Returns
    -------
    x : jax.Array
        Interpolated feature maps.
    """

    x = resize(x, (x.shape[0], round(x.shape[1] * scale), round(x.shape[2] * scale), x.shape[3]), *args, **kwargs)

    return x
