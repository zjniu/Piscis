import jax.numpy as jnp

from jax import vmap

from piscis.utils import apply_deltas


def spots_loss(deltas_pred, labels_pred, deltas, labels, dilated_labels, epsilon=1e-7, reduction='mean'):

    vmap_spots_loss = vmap(_spots_loss, in_axes=(0, 0, 0, 0, 0, None, None))
    rmse, bce, sf1 = vmap_spots_loss(deltas_pred, labels_pred, deltas, labels, dilated_labels, epsilon, reduction)
    rmse = _reduce_loss(rmse, reduction)
    bce = _reduce_loss(bce, reduction)
    sf1 = _reduce_loss(sf1, reduction)

    return rmse, bce, sf1


def _spots_loss(deltas_pred, labels_pred, deltas, labels, dilated_labels, epsilon, reduction):

    rmse = jnp.sqrt(jnp.sum(((deltas - deltas_pred) * dilated_labels) ** 2) / jnp.sum(dilated_labels))
    bce = weighted_bce_loss(labels_pred, labels, alpha=0.5, epsilon=epsilon, reduction=reduction)
    sf1 = smoothf1_loss(deltas_pred, labels_pred, labels, dilated_labels, epsilon=epsilon)

    return rmse, bce, sf1


def smoothf1_loss(deltas_pred, labels_pred, labels, dilated_labels, epsilon=1e-7):

    deltas_pred = deltas_pred * dilated_labels
    labels_pred = labels_pred[:, :, 0]
    labels = labels[:, :, 0]
    dilated_labels = dilated_labels[:, :, 0]

    counts = apply_deltas(deltas_pred, labels_pred, (3, 3))

    tp = jnp.sum(dilated_labels * counts)
    fp = jnp.sum(labels_pred) - tp

    num_captured = jnp.sum(labels_pred * labels)
    num_uncaptured = jnp.sum(labels) - num_captured
    colocalization_area = tp / (num_captured + epsilon)
    fn = num_uncaptured * colocalization_area

    sf1 = -2 * tp / (2 * tp + fp + fn + epsilon)

    return sf1


def dice_loss(y_pred, y, smooth: int = 1):

    intersection = jnp.sum(y_pred * y)
    dl = - (2.0 * intersection + smooth) / (jnp.sum(y_pred) + jnp.sum(y) + smooth)

    return dl


def mean_squared_error(y_pred, y, reduction='mean'):

    mse = (y - y_pred) ** 2
    mse = _reduce_loss(mse, reduction)

    return mse


def bce_loss(y_pred, y, pos_weight=1.0, epsilon=1e-7, reduction='mean'):

    bce = -(pos_weight * jnp.log(y_pred + epsilon) * y + jnp.log((1 - y_pred) + epsilon) * (1 - y))
    bce = _reduce_loss(bce, reduction)

    return bce


def bce_with_logits_loss(y_pred, y, pos_weight=1.0, reduction='mean'):

    neg_abs = -jnp.abs(y_pred)
    bce = jnp.maximum(y_pred, 0) - y_pred * y + jnp.log(1 + jnp.exp(neg_abs))
    bce = jnp.where(y, pos_weight * bce, bce)
    bce = _reduce_loss(bce, reduction)

    return bce


def weighted_bce_loss(y_pred, y, alpha=1.0, epsilon=1e-7, reduction='mean'):

    pos_weight = _inverse_class_weight(y, alpha=alpha, epsilon=epsilon)
    bce = bce_loss(y_pred, y, pos_weight=pos_weight, epsilon=epsilon, reduction=reduction)

    return bce


def weighted_bce_with_logits_loss(y_pred, y, alpha=1.0, epsilon=1e-7, reduction='mean'):

    pos_weight = _inverse_class_weight(y, alpha=alpha, epsilon=epsilon)
    bce = bce_with_logits_loss(y_pred, y, pos_weight=pos_weight, reduction=reduction)

    return bce


def cb_bce_loss(y_pred, y, beta=0.999, epsilon=1e-7, reduction='mean'):

    pos_weight = _class_balanced_weight(y, beta=beta)
    bce = bce_loss(y_pred, y, pos_weight=pos_weight, epsilon=epsilon, reduction=reduction)

    return bce


def cb_bce_with_logit_loss(y_pred, y, beta=0.999, reduction='mean'):

    pos_weight = _class_balanced_weight(y, beta=beta)
    bce = bce_with_logits_loss(y_pred, y, pos_weight=pos_weight, reduction=reduction)

    return bce


def la_loss(y_pred, y, tau=1.0, reduction='mean'):

    y_pred = _logit_adjustment(y_pred, y, tau=tau)
    la = bce_with_logits_loss(y_pred, y, reduction=reduction)

    return la


def vs_loss(y_pred, y, tau=1.0, gamma=0.25, reduction='mean'):

    y_pred = _vector_scaling(y_pred, y, tau=tau, gamma=gamma)
    vs = bce_with_logits_loss(y_pred, y, reduction=reduction)

    return vs


def binary_focal_loss(y_pred, y, gamma=2, epsilon=1e-7, reduction='mean'):

    bf = -((1 - y_pred) ** gamma * jnp.log(y_pred + epsilon) * y +
           y_pred ** gamma * jnp.log((1 - y_pred) + epsilon) * (1 - y))
    bf = _reduce_loss(bf, reduction=reduction)

    return bf


def cross_entropy_loss(y_pred, y, epsilon=1e-7, reduction='mean'):

    ce = -(jnp.log(y_pred + epsilon) * y)
    ce = _reduce_loss(ce, reduction=reduction)

    return ce


def focal_loss(labels_pred, labels, gamma=2, reduction='mean'):

    f = -((1 - labels_pred) ** gamma * jnp.log(labels_pred + 1e-7) * labels)
    f = _reduce_loss(f, reduction=reduction)

    return f


def _inverse_class_weight(y, alpha, epsilon):

    pos_weight = (jnp.sum(~y) / (jnp.sum(y) + epsilon)) ** alpha

    return pos_weight


def _class_balanced_weight(y, beta):

    pos_weight = (1 - beta ** jnp.sum(~y)) / (1 - beta ** jnp.sum(y))

    return pos_weight


def _logit_adjustment(y_pred, y, tau):

    num_pos = jnp.sum(y)
    num_neg = jnp.sum(~y)
    num_tot = num_pos + num_neg

    iota_pos = tau * jnp.log(num_pos / num_tot)
    iota_neg = tau * jnp.log(num_neg / num_tot)

    y_pred = jnp.where(y, y_pred + iota_pos, y_pred + iota_neg)

    return y_pred


def _vector_scaling(y_pred, y, tau, gamma):

    num_pos = jnp.sum(y)
    num_neg = jnp.sum(~y)
    num_tot = num_pos + num_neg
    num_max = jnp.maximum(num_pos, num_neg)

    iota_pos = tau * jnp.log(num_pos / num_tot)
    iota_neg = tau * jnp.log(num_neg / num_tot)
    delta_pos = (num_pos / num_max) ** gamma
    delta_neg = (num_neg / num_max) ** gamma

    y_pred = jnp.where(y, delta_pos * y_pred + iota_pos, delta_neg * y_pred + iota_neg)

    return y_pred


def _reduce_loss(loss, reduction='mean'):

    if reduction == 'mean':
        loss = jnp.mean(loss)
    elif reduction == 'sum':
        loss = jnp.sum(loss)

    return loss
