import jax
import jax.numpy as jnp

from functools import wraps
from jax import vmap
from typing import Callable, Optional, Union

from piscis.utils import smooth_sum_pool


def smoothf1_loss(
        deltas_pred: jax.Array,
        labels_pred: jax.Array,
        deltas: jax.Array,
        labels: jax.Array,
        dilation_iterations: int,
        max_distance: float,
        epsilon: float = 1e-7
) -> jax.Array:

    """Compute the SmoothF1 loss.

    Parameters
    ----------
    deltas_pred : jax.Array
        Predicted displacement vectors.
    labels_pred : jax.Array
        Predicted binary labels.
    deltas : jax.Array
        Ground truth displacement vectors.
    labels : jax.Array
        Ground truth binary labels.
    dilation_iterations : int
        Number of iterations used to dilate ground truth labels.
    max_distance : float
        Maximum distance for matching predicted and ground truth displacement vectors.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.

    Returns
    -------
    smoothf1 : jax.Array
        SmoothF1 loss.
    """

    # Use labels as the support for deltas_pred.
    deltas_pred = labels * deltas_pred

    # Squeeze the channel dimension in labels arrays.
    labels_pred = labels_pred[:, :, 0]
    labels = labels[:, :, 0]

    # Apply deltas_pred to labels_pred.
    kernel_size = (2 * dilation_iterations + 1, ) * 2
    pooled_labels = smooth_sum_pool(deltas_pred, labels_pred, 0.5, kernel_size)
    distances = jnp.linalg.norm(deltas_pred - deltas, axis=-1)
    matches = jnp.maximum(1 - distances / max_distance, 0.0)

    # Estimate the number of true positives, false positives, and false negatives.
    tp = jnp.sum(pooled_labels * matches)
    fp = jnp.sum(pooled_labels) - tp
    fn = jnp.sum(labels) - tp

    # Compute the SmoothF1 loss.
    smoothf1 = -2 * tp / (2 * tp + fp + fn + epsilon)

    return smoothf1


def dice_loss(
        y_pred: jax.Array,
        y: jax.Array,
        epsilon: float = 1e-7
) -> jax.Array:

    """Compute the Dice loss.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted binary labels.
    y : jax.Array
        Ground truth binary labels.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.

    Returns
    -------
    dl : jax.Array
        Dice loss.
    """

    intersection = jnp.sum(y_pred * y)
    dl = - (2.0 * intersection) / (jnp.sum(y_pred) + jnp.sum(y) + epsilon)

    return dl


def masked_l2_loss(
        y_pred: jax.Array,
        y: jax.Array,
        mask: jax.Array
) -> jax.Array:

    """Compute the L2 loss over masked pixels.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted values.
    y : jax.Array
        Ground truth values.
    mask : jax.Array
        Binary mask where 1 indicates pixels to compute the L2 loss and 0 indicates pixels to ignore.

    Returns
    -------
    rmse : jax.Array
        Masked root-mean-square error.
    """

    l2 = jnp.sum(jnp.linalg.norm(y_pred - y, axis=-1) * mask) / jnp.sum(mask)

    return l2


def mse_loss(
        y_pred: jax.Array,
        y: jax.Array,
) -> jax.Array:

    """Compute the mean squared error.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted values.
    y : jax.Array
        Ground truth values.

    Returns
    -------
    mse : jax.Array
        Mean squared error.
    """

    mse = (y - y_pred) ** 2
    mse = reduce_loss(mse, 'mean')

    return mse


def bce_loss(
        y_pred: jax.Array,
        y: jax.Array,
        pos_weight: Union[float, jax.Array] = 1.0,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jax.Array:

    """Compute the binary cross entropy loss.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted binary labels.
    y : jax.Array
        Ground truth binary labels.
    pos_weight : Union[float, jax.Array], optional
        Weight for positive class. Default is 1.0.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jax.Array
        Binary cross entropy loss.
    """

    bce = -(pos_weight * jnp.log(y_pred + epsilon) * y + jnp.log((1 - y_pred) + epsilon) * (1 - y))
    bce = reduce_loss(bce, reduction)

    return bce


def bce_with_logits_loss(
        y_pred: jax.Array,
        y: jax.Array,
        pos_weight: Union[float, jax.Array] = 1.0,
        reduction: Optional[str] = 'mean'
) -> jax.Array:

    """Compute the binary cross entropy loss with logits.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted logits.
    y : jax.Array
        Ground truth binary labels.
    pos_weight : Union[float, jax.Array], optional
        Weight for positive class. Default is 1.0.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jax.Array
        Binary cross entropy loss.
    """

    neg_abs = -jnp.abs(y_pred)
    bce = jnp.maximum(y_pred, 0) - y_pred * y + jnp.log(1 + jnp.exp(neg_abs))
    bce = jnp.where(y, pos_weight * bce, bce)
    bce = reduce_loss(bce, reduction)

    return bce


def weighted_bce_loss(
        y_pred: jax.Array,
        y: jax.Array,
        alpha: float = 1.0,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jax.Array:

    """Compute the weighted binary cross entropy loss.

    y_pred : jax.Array
        Predicted binary labels.
    y : jax.Array
        Ground truth binary labels.
    alpha : float, optional
        Exponent factor applied to the inverse class weight. Default is 1.0.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jax.Array
        Weighted binary cross entropy loss.
    """

    pos_weight = _inverse_class_weight(y, alpha=alpha, epsilon=epsilon)
    bce = bce_loss(y_pred, y, pos_weight=pos_weight, epsilon=epsilon, reduction=reduction)

    return bce


def weighted_bce_with_logits_loss(
        y_pred: jax.Array,
        y: jax.Array,
        alpha: float = 1.0,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jax.Array:

    """Compute the weighted binary cross entropy loss with logits.

    y_pred : jax.Array
        Predicted logits.
    y : jax.Array
        Ground truth binary labels.
    alpha : float, optional
        Exponent factor applied to the inverse class weight. Default is 1.0.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jax.Array
        Weighted binary cross entropy loss.
    """

    pos_weight = _inverse_class_weight(y, alpha=alpha, epsilon=epsilon)
    bce = bce_with_logits_loss(y_pred, y, pos_weight=pos_weight, reduction=reduction)

    return bce


def cb_bce_loss(
        y_pred: jax.Array,
        y: jax.Array,
        beta: float = 0.999,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jax.Array:

    """Compute the class-balanced binary cross entropy loss.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted binary labels.
    y : jax.Array
        Ground truth binary labels.
    beta : float, optional
        Hyperparameter for computing the effective number of samples. Default is 0.999.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jax.Array
        Class-balanced binary cross entropy loss.

    References
    ----------
    .. [1] Cui, Yin, et al. "Class-balanced loss based on effective number of samples." Proceedings of the IEEE/CVF
           conference on computer vision and pattern recognition. 2019.
    """

    pos_weight = _class_balanced_weight(y, beta=beta)
    bce = bce_loss(y_pred, y, pos_weight=pos_weight, epsilon=epsilon, reduction=reduction)

    return bce


def cb_bce_with_logit_loss(
        y_pred: jax.Array,
        y: jax.Array,
        beta: float = 0.999,
        reduction: Optional[str] = 'mean'
) -> jax.Array:

    """Compute the class-balanced binary cross entropy loss with logits.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted logits.
    y : jax.Array
        Ground truth binary labels.
    beta : float, optional
        Hyperparameter for computing the effective number of samples. Default is 0.999.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jax.Array
        Class-balanced binary cross entropy loss.

    References
    ----------
    .. [1] Cui, Yin, et al. "Class-balanced loss based on effective number of samples." Proceedings of the IEEE/CVF
           conference on computer vision and pattern recognition. 2019.
    """

    pos_weight = _class_balanced_weight(y, beta=beta)
    bce = bce_with_logits_loss(y_pred, y, pos_weight=pos_weight, reduction=reduction)

    return bce


def la_loss(
        y_pred: jax.Array,
        y: jax.Array,
        tau: float = 1.0,
        reduction: Optional[str] = 'mean'
):

    """Compute the logit-adjusted loss.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted logits.
    y : jax.Array
        Ground truth binary labels.
    tau : float, optional
        Hyperparameter for computing the logit adjustment. Default is 1.0.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    la : jax.Array
        Logit-adjusted loss.

    References
    ----------
    .. [1] Menon, Aditya Krishna, et al. "Long-tail learning via logit adjustment." arXiv preprint arXiv:2007.07314
           (2020).
    """

    y_pred = _logit_adjustment(y_pred, y, tau=tau)
    la = bce_with_logits_loss(y_pred, y, reduction=reduction)

    return la


def vs_loss(
        y_pred: jax.Array,
        y: jax.Array,
        tau: float = 1.0,
        gamma: float = 0.25,
        reduction: Optional[str] = 'mean'
):

    """Compute the vector-scaling loss.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted logits.
    y : jax.Array
        Ground truth binary labels.
    tau : float, optional
        Hyperparameter for computing the additive logit adjustment. Default is 1.0.
    gamma : float, optional
        Hyperparameter for computing the multiplicative logit adjustment. Default is 0.25.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    vs : jax.Array
        Vector-scaling loss.

    References
    ----------
    .. [1] Kini, Ganesh Ramachandra, et al. "Label-imbalanced and group-sensitive classification under
           overparameterization." Advances in Neural Information Processing Systems 34 (2021): 18970-18983.
    """

    y_pred = _vector_scaling(y_pred, y, tau=tau, gamma=gamma)
    vs = bce_with_logits_loss(y_pred, y, reduction=reduction)

    return vs


def binary_focal_loss(
        y_pred: jax.Array,
        y: jax.Array,
        gamma: float = 2,
        pos_weight: Union[float, jax.Array] = 1.0,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jax.Array:

    """Compute the binary focal loss.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted binary labels.
    y : jax.Array
        Ground truth binary labels.
    gamma : float, optional
        Hyperparameter for computing the binary focal loss. Default is 2.
    pos_weight : Union[float, jax.Array], optional
        Weight for positive class. Default is 1.0.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bf : jax.Array
        Binary focal loss.

    References
    ----------
    .. [1] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international
           conference on computer vision. 2017.
    """

    bce = bce_loss(y_pred, y, pos_weight=pos_weight, epsilon=epsilon, reduction=reduction)
    p_t = y_pred * y + (1 - y_pred) * (1 - y)
    bf = bce * (1 - p_t) ** gamma
    bf = reduce_loss(bf, reduction=reduction)

    return bf


def weighted_binary_focal_loss(
        y_pred: jax.Array,
        y: jax.Array,
        alpha: float = 1.0,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jax.Array:

    """Compute the weighted binary focal loss.

    y_pred : jax.Array
        Predicted binary labels.
    y : jax.Array
        Ground truth binary labels.
    alpha : float, optional
        Exponent factor applied to the inverse class weight. Default is 1.0.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jax.Array
        Weighted binary focal loss.
    """

    pos_weight = _inverse_class_weight(y, alpha=alpha, epsilon=epsilon)
    bf = binary_focal_loss(y_pred, y, pos_weight=pos_weight, epsilon=epsilon, reduction=reduction)

    return bf


def ce_loss(
        y_pred: jax.Array,
        y: jax.Array,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jax.Array:

    """Compute the cross entropy loss.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted labels.
    y : jax.Array
        Ground truth labels.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    ce : jax.Array
        Cross entropy loss.
    """

    ce = -(jnp.log(y_pred + epsilon) * y)
    ce = reduce_loss(ce, reduction=reduction)

    return ce


def focal_loss(
        y_pred: jax.Array,
        y: jax.Array,
        gamma: float = 2,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jax.Array:

    """Compute the focal loss.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted labels.
    y : jax.Array
        Ground truth labels.
    gamma : float, optional
        Hyperparameter for computing the focal loss. Default is 2.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    f : jax.Array
        Focal loss.

    References
    ----------
    .. [1] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international
           conference on computer vision. 2017.
    """

    f = -((1 - y_pred) ** gamma * jnp.log(y_pred + epsilon) * y)
    f = reduce_loss(f, reduction=reduction)

    return f


def _inverse_class_weight(
        y: jax.Array,
        alpha: float,
        epsilon: float
) -> jax.Array:

    """Compute the inverse class weight.

    Parameters
    ----------
    y : jax.Array
        Ground truth binary labels.
    alpha : float
        Exponent of the inverse class weight.
    epsilon : float
        Small constant for numerical stability.

    Returns
    -------
    pos_weight : jax.Array
        Inverse class weight.
    """

    pos_weight = (jnp.sum(~y) / (jnp.sum(y) + epsilon)) ** alpha

    return pos_weight


def _class_balanced_weight(
        y: jax.Array,
        beta: float
) -> jax.Array:

    """Compute the class-balanced weight.

    Parameters
    ----------
    y : jax.Array
        Ground truth binary labels.
    beta : float
        Hyperparameter for computing the effective number of samples.

    Returns
    -------
    pos_weight : jax.Array
        Class-balanced weight.

    References
    ----------
    .. [1] Cui, Yin, et al. "Class-balanced loss based on effective number of samples." Proceedings of the IEEE/CVF
           conference on computer vision and pattern recognition. 2019.
    """

    pos_weight = (1 - beta ** jnp.sum(~y)) / (1 - beta ** jnp.sum(y))

    return pos_weight


def _logit_adjustment(
        y_pred: jax.Array,
        y: jax.Array,
        tau: float
) -> jax.Array:

    """Apply logit adjustment to the predicted logits.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted logits.
    y : jax.Array
        Ground truth binary labels.
    tau : float
        Hyperparameter for computing the logit adjustment.

    Returns
    -------
    y_pred : jax.Array
        Adjusted logits.

    References
    ----------
    .. [1] Menon, Aditya Krishna, et al. "Long-tail learning via logit adjustment." arXiv preprint arXiv:2007.07314
           (2020).
    """

    # Compute class sizes.
    num_pos = jnp.sum(y)
    num_neg = jnp.sum(~y)
    num_tot = num_pos + num_neg

    # Compute logit adjustments.
    iota_pos = tau * jnp.log(num_pos / num_tot)
    iota_neg = tau * jnp.log(num_neg / num_tot)

    # Apply logit adjustments.
    y_pred = jnp.where(y, y_pred + iota_pos, y_pred + iota_neg)

    return y_pred


def _vector_scaling(
        y_pred: jax.Array,
        y: jax.Array,
        tau: float,
        gamma: float
) -> jax.Array:

    """Apply vector scaling to the predicted logits.

    Parameters
    ----------
    y_pred : jax.Array
        Predicted logits.
    y : jax.Array
        Ground truth binary labels.
    tau : float
        Hyperparameter for computing the additive logit adjustment.
    gamma : float
        Hyperparameter for computing the multiplicative logit adjustment.

    Returns
    -------
    y_pred : jax.Array
        Adjusted logits.

    References
    ----------
    .. [1] Kini, Ganesh Ramachandra, et al. "Label-imbalanced and group-sensitive classification under
           overparameterization." Advances in Neural Information Processing Systems 34 (2021): 18970-18983.
    """

    # Compute class sizes.
    num_pos = jnp.sum(y)
    num_neg = jnp.sum(~y)
    num_tot = num_pos + num_neg
    num_max = jnp.maximum(num_pos, num_neg)

    # Compute additive logit adjustments.
    iota_pos = tau * jnp.log(num_pos / num_tot)
    iota_neg = tau * jnp.log(num_neg / num_tot)

    # Compute multiplicative logit adjustments.
    delta_pos = (num_pos / num_max) ** gamma
    delta_neg = (num_neg / num_max) ** gamma

    # Apply vector scaling.
    y_pred = jnp.where(y, delta_pos * y_pred + iota_pos, delta_neg * y_pred + iota_neg)

    return y_pred


def wrap_loss_fn(
        loss_fn: Callable,
        axis: int = 0,
        reduction: Optional[str] = 'mean'
) -> Callable:
    """Wrap a loss function for vectorization and loss reduction.

    Parameters
    ----------
    loss_fn : Callable
        Loss function.
    axis : int, optional
        Axis to vectorize over. Default is 0.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    wrapped_loss_fn : Callable
        Wrapped loss function.
    """

    @wraps(loss_fn)
    def wrapped_loss_fn(*args):

        # Build in_axes dynamically from the arguments.
        in_axes = tuple(axis if isinstance(arg, jax.Array) else None for arg in args)

        # Vectorize the loss function.
        loss = vmap(loss_fn, in_axes=in_axes)(*args)

        # Reduce the loss.
        loss = reduce_loss(loss, reduction=reduction)

        return loss

    return wrapped_loss_fn


def reduce_loss(
        loss: jax.Array,
        reduction: Optional[str] = 'mean'
) -> jax.Array:

    """Reduce the loss.

    Parameters
    ----------
    loss : jax.Array
        Loss array to be reduced.
    reduction : Optional[str], optional
        Loss reduction method. Supported methods are 'mean' and 'sum'. Default is 'mean'.

    Returns
    -------
    loss : jax.Array
        Reduced loss.

    Raises
    ------
    ValueError
        If the `reduction` method is not supported.
    """

    if reduction == 'mean':
        loss = jnp.mean(loss)
    elif reduction == 'sum':
        loss = jnp.sum(loss)
    else:
        raise ValueError("Reduction method is not supported.")

    return loss
