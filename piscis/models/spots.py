from flax import linen as nn

from piscis.networks.efficientnetv2 import build_efficientnetv2
from piscis.networks.fpn import FPN

blocks_args = [
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

EfficientNetV2XS = build_efficientnetv2(
    blocks_args=blocks_args,
    model_name='EfficientNetV2XS',
    width_coefficient=1.0,
    depth_coefficient=1.0,
    stem_strides=1,
    dropout_rate=0.2
)


class SpotsModel(nn.Module):

    style: bool = True
    aggregate: str = 'sum'

    @nn.compact
    def __call__(self, x, train: bool = True):

        x, style = FPN(
            encoder=EfficientNetV2XS,
            encoder_levels={0, 1, 2, 3},
            features=3,
            style=self.style,
            aggregate=self.aggregate
        )(x, train=train)
        deltas = x[:, :, :, :2]
        labels = nn.sigmoid(x[:, :, :, 2:3])

        return deltas, labels
