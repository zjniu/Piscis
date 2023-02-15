import jax.numpy as np

from jax import vmap

from piscis.utils import colocalize_pixels


def spots_loss(deltas_pred, labels_pred, deltas, labels, dilated_labels):

    vmap_colocalization_loss = vmap(_spots_loss, in_axes=(0, 0, 0, 0, 0))
    sl_rmse, sl_bcel, sl_smoothf1 = vmap_colocalization_loss(deltas_pred, labels_pred, deltas, labels, dilated_labels)

    return np.mean(sl_rmse), np.mean(sl_bcel), np.mean(sl_smoothf1)


def _spots_loss(deltas_pred, labels_pred, deltas, labels, dilated_labels):

    labels_pred = labels_pred[:, :, 0]
    labels = labels[:, :, 0]
    dilated_labels = dilated_labels[:, :, 0]

    sl_rmse = np.sqrt(np.sum(((deltas - deltas_pred) * dilated_labels[:, :, None]) ** 2) / np.sum(dilated_labels))
    sl_bcel = binary_cross_entropy_loss(labels_pred, labels, alpha=0.5)

    counts = colocalize_pixels(deltas_pred * dilated_labels[:, :, None], labels_pred, (3, 3))

    tp = np.sum(dilated_labels * counts)
    fp = np.sum(labels_pred) - tp

    num_captured = np.sum(labels_pred * labels)
    num_uncaptured = np.sum(labels) - num_captured
    colocalization_area = tp / (num_captured + 1e-07)
    fn = num_uncaptured * colocalization_area

    precision = tp / (tp + fp + 1e-07)
    recall = tp / (tp + fn + 1e-07)
    sl_smoothf1 = -2 * precision * recall / (precision + recall + 1e-07)

    return sl_rmse, sl_bcel, sl_smoothf1


def dice_loss(y_pred, y, smooth: int = 1):

    intersection = np.sum(y_pred * y)
    dl = - (2.0 * intersection + smooth) / (np.sum(y_pred) + np.sum(y) + smooth)

    return dl


def mean_squared_error(y_pred, y):

    mse = np.mean((y - y_pred) ** 2)

    return mse


def binary_cross_entropy_loss(labels_pred, labels, alpha=0.0, epsilon=1e-7):

    if alpha == 0:
        beta = 1
    else:
        beta = (np.sum(~labels) / (np.sum(labels) + epsilon)) ** alpha

    bcel = -np.mean(beta * np.log(labels_pred + 1e-7) * labels + np.log((1 - labels_pred) + 1e-7) * (1 - labels))

    return bcel


def binary_focal_loss(labels_pred, labels, gamma=2, alpha=0.0, epsilon=1e-7):

    if alpha == 0:
        beta = 1
    else:
        beta = (np.sum(~labels) / (np.sum(labels) + epsilon)) ** alpha

    bfl = -np.mean(beta * (1 - labels_pred) ** gamma * np.log(labels_pred + 1e-7) * labels +
                   labels_pred ** gamma * np.log((1 - labels_pred) + 1e-7) * (1 - labels))

    return bfl


def cross_entropy_loss(labels_pred, labels, alpha=0):

    if alpha > 0:
        beta = _calculate_class_weights(labels, alpha).reshape(1, 1, 1, -1)
    else:
        beta = 1

    cel = -np.mean(beta * np.log(labels_pred + 1e-7) * labels)

    return cel


def focal_loss(labels_pred, labels, gamma=2, alpha=0):
    if alpha > 0:
        beta = _calculate_class_weights(labels, alpha).reshape(1, 1, -1)
    else:
        beta = 1

    fl = -np.mean(beta * (1 - labels_pred) ** gamma * np.log(labels_pred + 1e-7) * labels)

    return fl


def _calculate_class_weights(labels, alpha):

    class_sums = np.sum(labels, axis=(0, 1, 2))
    class_sums = (1 / (class_sums + 1)) ** alpha
    beta = class_sums / class_sums.sum()

    return beta
