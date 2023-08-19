import jax.numpy as jnp

from jax import vmap
from typing import Optional, Tuple, Union

from piscis.utils import smooth_sum_pool


def spots_loss(
        deltas_pred: jnp.ndarray,
        labels_pred: jnp.ndarray,
        deltas: jnp.ndarray,
        labels: jnp.ndarray,
        dilated_labels: jnp.ndarray,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    """Compute the loss function for training SpotsModel.

    Parameters
    ----------
    deltas_pred : jnp.ndarray
        Predicted subpixel displacements.
    labels_pred : jnp.ndarray
        Predicted binary labels.
    deltas : jnp.ndarray
        Ground truth subpixel displacements.
    labels : jnp.ndarray
        Ground truth binary labels.
    dilated_labels : jnp.ndarray
        Dilated ground truth binary labels.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    rmse : jnp.ndarray
        Root mean squared error.
    bce : jnp.ndarray
        Binary cross entropy loss.
    sf1 : jnp.ndarray
        SmoothF1 loss.
    """

    # Vectorize the loss function.
    vmap_spots_loss = vmap(_spots_loss, in_axes=(0, 0, 0, 0, 0, None, None))

    # Compute and reduce loss terms.
    rmse, bce, sf1 = vmap_spots_loss(deltas_pred, labels_pred, deltas, labels, dilated_labels, epsilon, reduction)
    rmse = _reduce_loss(rmse, reduction)
    bce = _reduce_loss(bce, reduction)
    sf1 = _reduce_loss(sf1, reduction)

    return rmse, bce, sf1


def _spots_loss(
        deltas_pred: jnp.ndarray,
        labels_pred: jnp.ndarray,
        deltas: jnp.ndarray,
        labels: jnp.ndarray,
        dilated_labels: jnp.ndarray,
        epsilon: float,
        reduction: Optional[str]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    """Compute the rmse, bce, and sf1 loss terms for a single batch element.

    Parameters
    ----------
    deltas_pred : jnp.ndarray
        Predicted subpixel displacements.
    labels_pred : jnp.ndarray
        Predicted binary labels.
    deltas : jnp.ndarray
        Ground truth subpixel displacements.
    labels : jnp.ndarray
        Ground truth binary labels.
    dilated_labels : jnp.ndarray
        Dilated ground truth binary labels.
    epsilon : float
        Small constant for numerical stability.
    reduction : Optional[str]
        Loss reduction method.

    Returns
    -------
    rmse : jnp.ndarray
        Root mean squared error.
    bce : jnp.ndarray
        Binary cross entropy loss.
    sf1 : jnp.ndarray
        SmoothF1 loss.
    """

    rmse = jnp.sqrt(jnp.sum(((deltas - deltas_pred) * dilated_labels) ** 2) / jnp.sum(dilated_labels))
    bce = weighted_bce_loss(labels_pred, dilated_labels, alpha=0.5, epsilon=epsilon, reduction=reduction)
    sf1 = smoothf1_loss(deltas_pred, labels_pred, labels, dilated_labels, epsilon=epsilon)

    return rmse, bce, sf1


def smoothf1_loss(
        deltas_pred: jnp.ndarray,
        labels_pred: jnp.ndarray,
        labels: jnp.ndarray,
        dilated_labels: jnp.ndarray,
        epsilon: float = 1e-7
) -> jnp.ndarray:

    """Compute the SmoothF1 loss for a single batch element.

    Parameters
    ----------
    deltas_pred : jnp.ndarray
        Predicted subpixel displacements.
    labels_pred : jnp.ndarray
        Predicted binary labels.
    labels : jnp.ndarray
        Ground truth binary labels.
    dilated_labels : jnp.ndarray
        Dilated ground truth binary labels.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.

    Returns
    -------
    sf1 : jnp.ndarray
        SmoothF1 loss.
    """

    # Use dilated_labels as the support for deltas_pred.
    deltas_pred = deltas_pred * dilated_labels

    # Squeeze the channel dimension in labels arrays.
    labels_pred = labels_pred[:, :, 0]
    labels = labels[:, :, 0]
    dilated_labels = dilated_labels[:, :, 0]

    # Apply deltas_pred to labels_pred.
    pooled_labels = smooth_sum_pool(deltas_pred, labels_pred, 0.5, (3, 3))

    # Estimate the number of true positives and false positives.
    tp = jnp.sum(dilated_labels * pooled_labels)
    fp = jnp.sum(pooled_labels) - tp

    # Estimate the mean mass of captured spots.
    num_captured = jnp.sum(labels_pred * labels)
    num_uncaptured = jnp.sum(labels) - num_captured
    spot_mass = tp / (num_captured + epsilon)

    # Estimate the number of false negatives.
    fn = num_uncaptured * spot_mass

    # Compute the SmoothF1 loss.
    sf1 = -2 * tp / (2 * tp + fp + fn + epsilon)

    return sf1


def dice_loss(
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        epsilon: float = 1e-7
) -> jnp.ndarray:

    """Compute the Dice loss.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted binary labels.
    y : jnp.ndarray
        Ground truth binary labels.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.

    Returns
    -------
    dl : jnp.ndarray
        Dice loss.
    """

    intersection = jnp.sum(y_pred * y)
    dl = - (2.0 * intersection) / (jnp.sum(y_pred) + jnp.sum(y) + epsilon)

    return dl


def mean_squared_error(
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        reduction: Optional[str] = 'mean'
) -> jnp.ndarray:

    """Compute the mean squared error.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted values.
    y : jnp.ndarray
        Ground truth values.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    mse : jnp.ndarray
        Mean squared error.
    """

    mse = (y - y_pred) ** 2
    mse = _reduce_loss(mse, reduction)

    return mse


def bce_loss(
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        pos_weight: Union[float, jnp.ndarray] = 1.0,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jnp.ndarray:

    """Compute the binary cross entropy loss.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted binary labels.
    y : jnp.ndarray
        Ground truth binary labels.
    pos_weight : Union[float, jnp.ndarray], optional
        Weight for positive class. Default is 1.0.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jnp.ndarray
        Binary cross entropy loss.
    """

    bce = -(pos_weight * jnp.log(y_pred + epsilon) * y + jnp.log((1 - y_pred) + epsilon) * (1 - y))
    bce = _reduce_loss(bce, reduction)

    return bce


def bce_with_logits_loss(
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        pos_weight: Union[float, jnp.ndarray] = 1.0,
        reduction: Optional[str] = 'mean'
) -> jnp.ndarray:

    """Compute the binary cross entropy loss with logits.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted logits.
    y : jnp.ndarray
        Ground truth binary labels.
    pos_weight : Union[float, jnp.ndarray], optional
        Weight for positive class. Default is 1.0.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jnp.ndarray
        Binary cross entropy loss.
    """

    neg_abs = -jnp.abs(y_pred)
    bce = jnp.maximum(y_pred, 0) - y_pred * y + jnp.log(1 + jnp.exp(neg_abs))
    bce = jnp.where(y, pos_weight * bce, bce)
    bce = _reduce_loss(bce, reduction)

    return bce


def weighted_bce_loss(
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        alpha: float = 1.0,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jnp.ndarray:

    """Compute the weighted binary cross entropy loss.

    y_pred : jnp.ndarray
        Predicted binary labels.
    y : jnp.ndarray
        Ground truth binary labels.
    alpha : float, optional
        Exponent factor applied to the inverse class weight. Default is 1.0.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jnp.ndarray
        Weighted binary cross entropy loss.
    """

    pos_weight = _inverse_class_weight(y, alpha=alpha, epsilon=epsilon)
    bce = bce_loss(y_pred, y, pos_weight=pos_weight, epsilon=epsilon, reduction=reduction)

    return bce


def weighted_bce_with_logits_loss(
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        alpha: float = 1.0,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jnp.ndarray:

    """Compute the weighted binary cross entropy loss with logits.

    y_pred : jnp.ndarray
        Predicted logits.
    y : jnp.ndarray
        Ground truth binary labels.
    alpha : float, optional
        Exponent factor applied to the inverse class weight. Default is 1.0.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jnp.ndarray
        Weighted binary cross entropy loss.
    """

    pos_weight = _inverse_class_weight(y, alpha=alpha, epsilon=epsilon)
    bce = bce_with_logits_loss(y_pred, y, pos_weight=pos_weight, reduction=reduction)

    return bce


def cb_bce_loss(
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        beta: float = 0.999,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jnp.ndarray:

    """Compute the class-balanced binary cross entropy loss.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted binary labels.
    y : jnp.ndarray
        Ground truth binary labels.
    beta : float, optional
        Hyperparameter for computing the effective number of samples. Default is 0.999.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jnp.ndarray
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
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        beta: float = 0.999,
        reduction: Optional[str] = 'mean'
) -> jnp.ndarray:

    """Compute the class-balanced binary cross entropy loss with logits.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted logits.
    y : jnp.ndarray
        Ground truth binary labels.
    beta : float, optional
        Hyperparameter for computing the effective number of samples. Default is 0.999.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bce : jnp.ndarray
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
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        tau: float = 1.0,
        reduction: Optional[str] = 'mean'
):

    """Compute the logit-adjusted loss.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted logits.
    y : jnp.ndarray
        Ground truth binary labels.
    tau : float, optional
        Hyperparameter for computing the logit adjustment. Default is 1.0.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    la : jnp.ndarray
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
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        tau: float = 1.0,
        gamma: float = 0.25,
        reduction: Optional[str] = 'mean'
):

    """Compute the vector-scaling loss.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted logits.
    y : jnp.ndarray
        Ground truth binary labels.
    tau : float, optional
        Hyperparameter for computing the additive logit adjustment. Default is 1.0.
    gamma : float, optional
        Hyperparameter for computing the multiplicative logit adjustment. Default is 0.25.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    vs : jnp.ndarray
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
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        gamma: float = 2,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jnp.ndarray:

    """Compute the binary focal loss.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted binary labels.
    y : jnp.ndarray
        Ground truth binary labels.
    gamma : float, optional
        Hyperparameter for computing the binary focal loss. Default is 2.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    bf : jnp.ndarray
        Binary focal loss.

    References
    ----------
    .. [1] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international
           conference on computer vision. 2017.
    """

    bf = -((1 - y_pred) ** gamma * jnp.log(y_pred + epsilon) * y +
           y_pred ** gamma * jnp.log((1 - y_pred) + epsilon) * (1 - y))
    bf = _reduce_loss(bf, reduction=reduction)

    return bf


def cross_entropy_loss(
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jnp.ndarray:

    """Compute the cross entropy loss.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted labels.
    y : jnp.ndarray
        Ground truth labels.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    ce : jnp.ndarray
        Cross entropy loss.
    """

    ce = -(jnp.log(y_pred + epsilon) * y)
    ce = _reduce_loss(ce, reduction=reduction)

    return ce


def focal_loss(
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        gamma: float = 2,
        epsilon: float = 1e-7,
        reduction: Optional[str] = 'mean'
) -> jnp.ndarray:

    """Compute the focal loss.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted labels.
    y : jnp.ndarray
        Ground truth labels.
    gamma : float, optional
        Hyperparameter for computing the focal loss. Default is 2.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    f : jnp.ndarray
        Focal loss.

    References
    ----------
    .. [1] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international
           conference on computer vision. 2017.
    """

    f = -((1 - y_pred) ** gamma * jnp.log(y_pred + epsilon) * y)
    f = _reduce_loss(f, reduction=reduction)

    return f


def _inverse_class_weight(
        y: jnp.ndarray,
        alpha: float,
        epsilon: float
) -> jnp.ndarray:

    """Compute the inverse class weight.

    Parameters
    ----------
    y : jnp.ndarray
        Ground truth binary labels.
    alpha : float
        Exponent of the inverse class weight.
    epsilon : float
        Small constant for numerical stability.

    Returns
    -------
    pos_weight : jnp.ndarray
        Inverse class weight.
    """

    pos_weight = (jnp.sum(~y) / (jnp.sum(y) + epsilon)) ** alpha

    return pos_weight


def _class_balanced_weight(
        y: jnp.ndarray,
        beta: float
) -> jnp.ndarray:

    """Compute the class-balanced weight.

    Parameters
    ----------
    y : jnp.ndarray
        Ground truth binary labels.
    beta : float
        Hyperparameter for computing the effective number of samples.

    Returns
    -------
    pos_weight : jnp.ndarray
        Class-balanced weight.

    References
    ----------
    .. [1] Cui, Yin, et al. "Class-balanced loss based on effective number of samples." Proceedings of the IEEE/CVF
           conference on computer vision and pattern recognition. 2019.
    """

    pos_weight = (1 - beta ** jnp.sum(~y)) / (1 - beta ** jnp.sum(y))

    return pos_weight


def _logit_adjustment(
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        tau: float
) -> jnp.ndarray:

    """Apply logit adjustment to the predicted logits.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted logits.
    y : jnp.ndarray
        Ground truth binary labels.
    tau : float
        Hyperparameter for computing the logit adjustment.

    Returns
    -------
    y_pred : jnp.ndarray
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
        y_pred: jnp.ndarray,
        y: jnp.ndarray,
        tau: float,
        gamma: float
) -> jnp.ndarray:

    """Apply vector scaling to the predicted logits.

    Parameters
    ----------
    y_pred : jnp.ndarray
        Predicted logits.
    y : jnp.ndarray
        Ground truth binary labels.
    tau : float
        Hyperparameter for computing the additive logit adjustment.
    gamma : float
        Hyperparameter for computing the multiplicative logit adjustment.

    Returns
    -------
    y_pred : jnp.ndarray
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


def _reduce_loss(
        loss: jnp.ndarray,
        reduction: Optional[str] = 'mean'
) -> jnp.ndarray:

    """Reduce the loss.

    Parameters
    ----------
    loss : jnp.ndarray
        Loss array to be reduced.
    reduction : Optional[str], optional
        Loss reduction method. Default is 'mean'.

    Returns
    -------
    loss : jnp.ndarray
        Reduced loss.
    """

    if reduction == 'mean':
        loss = jnp.mean(loss)
    elif reduction == 'sum':
        loss = jnp.sum(loss)

    return loss
