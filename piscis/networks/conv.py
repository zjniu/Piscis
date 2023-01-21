import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from typing import Any, Callable, Sequence, Union

ModuleDef = Any


class Conv(nn.Module):

    features: int
    kernel_size: Sequence[int]
    conv: ModuleDef
    bn: ModuleDef
    act: Callable | None
    layers: Sequence[str]

    @nn.compact
    def __call__(self, x):

        conv = partial(
            self.conv,
            features=self.features,
            kernel_size=self.kernel_size,
        )

        layers = []

        for layer in self.layers:
            if layer == 'conv':
                layers.append(conv())
            elif layer == 'bn':
                layers.append(self.bn())
            elif layer == 'act':
                layers.append(self.act)
            else:
                raise ValueError(f'Unknown layer {layer}.')

        x = nn.Sequential(layers)(x)

        return x


class MBConv(nn.Module):

    features_in: int
    features_out: int
    expand_ratio: int
    kernel_size: Sequence[int]
    strides: int
    se_ratio: float
    conv: ModuleDef
    dropout: Union[None, ModuleDef]
    bn: ModuleDef
    act: Callable

    @nn.compact
    def __call__(self, x):

        conv = partial(
            self.conv,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME'
        )

        proj = partial(
            self.conv,
            kernel_size=(1, 1),
            strides=1,
            padding='SAME'
        )

        inputs = x

        # Expansion phase
        filters = self.features_in * self.expand_ratio
        if self.expand_ratio != 1:

            x = proj(
                features=filters,
                use_bias=False,
                name='expand_conv',
            )(x)
            x = self.bn(
                name='expand_bn'
            )(x)
            x = self.act(x)

        # Depthwise conv
        x = conv(
            features=x.shape[-1],
            feature_group_count=x.shape[-1],
            use_bias=False,
            name='dw_conv'
        )(x)
        x = self.bn(
            name='dw_bn'
        )(x)
        x = self.act(x)

        # Squeeze and excite
        if 0 < self.se_ratio <= 1:

            filters_se = max(1, int(self.features_in * self.se_ratio))
            se = jnp.mean(x, axis=(1, 2), keepdims=True)

            se = proj(
                features=filters_se,
                use_bias=True,
                name='se_reduce'
            )(se)
            se = self.act(se)
            se = proj(
                features=filters,
                use_bias=True,
                name='se_expand'
            )(se)
            se = nn.sigmoid(se)

            x = x * se

        # Output phase
        x = proj(
            features=self.features_out,
            use_bias=False,
            name='project_conv'
        )(x)
        x = self.bn(
            name='project_bn'
        )(x)

        # Residual
        if (self.strides == 1) and (self.features_in == self.features_out):

            if self.dropout is not None:
                x = self.dropout(
                    name='drop'
                )(x)
            x = x + inputs

        return x


class FusedMBConv(nn.Module):

    features_in: int
    features_out: int
    expand_ratio: int
    kernel_size: Sequence[int]
    strides: int
    se_ratio: float
    conv: ModuleDef
    dropout: Union[None, ModuleDef]
    bn: ModuleDef
    act: Callable

    @nn.compact
    def __call__(self, x):

        conv = partial(
            self.conv,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='SAME'
        )

        proj = partial(
            self.conv,
            kernel_size=(1, 1),
            strides=1,
            padding='SAME'
        )

        inputs = x

        # Expansion phase
        filters = self.features_in * self.expand_ratio
        if self.expand_ratio != 1:

            x = conv(
                features=filters,
                use_bias=False,
                name='expand_conv',
            )(x)
            x = self.bn(
                name='expand_bn'
            )(x)
            x = self.act(x)

        # Squeeze and excite
        if 0 < self.se_ratio <= 1:

            filters_se = max(1, int(self.features_in * self.se_ratio))
            se = jnp.mean(x, axis=(1, 2), keepdims=True)

            se = proj(
                features=filters_se,
                use_bias=True,
                name='se_reduce'
            )(se)
            se = self.act(se)
            se = proj(
                features=filters,
                use_bias=True,
                name='se_expand'
            )(se)
            se = nn.sigmoid(se)

            x = x * se

        # Output phase
        if self.expand_ratio == 1:
            x = conv(
                features=self.features_out,
                use_bias=False,
                name='project_conv'
            )(x)
        else:
            x = proj(
                features=self.features_out,
                use_bias=False,
                name='project_conv'
            )(x)
        x = self.bn(
            name='project_bn'
        )(x)
        if self.expand_ratio == 1:
            x = self.act(x)

        # Residual
        if (self.strides == 1) and (self.features_in == self.features_out):

            if self.dropout is not None:
                x = self.dropout(
                    name='drop'
                )(x)
            x = x + inputs

        return x
