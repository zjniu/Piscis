import optax

from typing import Any, Callable, Optional, Union


def _scale_by_learning_rate(learning_rate: optax.ScalarOrSchedule, flip_sign=True):
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return optax.scale_by_schedule(lambda count: m * learning_rate(count))
    return optax.scale(m * learning_rate)


def adabelief(
    learning_rate: optax.ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-16,
    eps_root: float = 1e-16,
    weight_decay: float = 1e-4,
    mask: Optional[Union[Any, Callable[[optax.Params], Any]]] = None
) -> optax.GradientTransformation:
    """The AdaBelief optimizer.
    AdaBelief is an adaptive learning rate optimizer that focuses on fast
    convergence, generalization, and stability. It adapts the step size depending
    on its "belief" in the gradient direction — the optimizer adaptively scales
    the step size by the difference between the predicted and observed gradients.
    AdaBelief is a modified version of Adam and contains the same number of
    parameters.
    References:
        Zhuang et al, 2020: https://arxiv.org/abs/2010.07468
    Args:
        learning_rate: A fixed global scaling factor.
        b1: Exponential decay rate to track the first moment of past gradients.
        b2: Exponential decay rate to track the second moment of past gradients.
        eps: Term added to the denominator to improve numerical stability.
        eps_root: Term added to the second moment of the prediction error to
            improve numerical stability. If backpropagating gradients through the
            gradient transformation (e.g. for meta-learning), this must be non-zero.
        weight_decay: Strength of the weight decay regularization. Note that this
            weight decay is multiplied with the learning rate. This is consistent
            with other frameworks such as PyTorch, but different from
            (Loshchilov et al, 2019) where the weight decay is only multiplied with
            the "schedule multiplier", but not the base learning rate.
        mask: A tree with same structure as (or a prefix of) the params PyTree,
            or a Callable that returns such a pytree given the params/updates.
            The leaves should be booleans, `True` for leaves/subtrees you want to
            apply the weight decay to, and `False` for those you want to skip. Note
            that the Adabelief gradient transformations are applied to all parameters.
    Returns:
        The corresponding `GradientTransformation`.
    """
    return optax.chain(
        optax.scale_by_belief(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        optax.add_decayed_weights(weight_decay, mask),
        _scale_by_learning_rate(learning_rate)
    )
