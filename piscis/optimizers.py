import optax

from typing import Any, Callable, Optional, Union


def sgdw(
    learning_rate: optax.ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None,
) -> optax.GradientTransformation:

    """A canonical Stochastic Gradient Descent optimizer.

    This implements stochastic gradient descent. It also includes support for momentum, and nesterov acceleration, as
    these are standard practice when using stochastic gradient descent to train deep neural networks.

    Parameters
    ----------
    learning_rate : Union[optax.Schedule, float]
        A fixed global scaling factor.
    momentum : Optional[float], optional
        Decay rate used by the momentum term. If None, momentum is not used at all. Default is None.
    nesterov : bool, optional
        Whether Nesterov momentum is used. Default is False.
    accumulator_dtype : Optional[Any], optional
        Optional `dtype` to be used for the accumulator. If None, the `dtype` is inferred from `params` and `updates`.
        Default is None.
    weight_decay : float, optional
        Strength of the weight decay regularization. Note that this weight decay is multiplied with the learning rate.
        This is consistent with other frameworks such as PyTorch, but different from (Loshchilov et al., 2019) where the
        weight decay is only multiplied with the "schedule multiplier", but not the base learning rate.
    mask : Optional[Union[Any, Callable[[optax.Params], Any]]], optional
        A tree with same structure as (or a prefix of) the params PyTree, or a Callable that returns such a pytree given
        the params/updates. The leaves should be booleans, True for leaves/subtrees you want to apply the weight decay
        to, and False for those you want to skip. Note that the SGDW gradient transformations are applied to all
        parameters.

    Returns
    -------
    tx : optax.GradientTransformation
        SGDW gradient transformation.

    References
    ----------
    .. [1] Sutskever, Ilya, et al. "On the importance of initialization and momentum in deep learning." International
           conference on machine learning. PMLR, 2013.
    .. [2] Ilya, Loshchilov, and Hutter Frank. "Decoupled weight decay regularization." Proceedings of ICLR 7 (2019).
    """

    tx = optax.chain(
        (
            optax.trace(
                decay=momentum, nesterov=nesterov, accumulator_dtype=accumulator_dtype
            )
            if momentum is not None
            else optax.identity()
        ),
        optax.add_decayed_weights(weight_decay, mask),
        optax._src.alias._scale_by_learning_rate(learning_rate),
    )

    return tx
