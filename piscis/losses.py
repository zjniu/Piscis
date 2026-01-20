import torch

from functools import wraps
from typing import Callable, Optional, Sequence

from piscis.utils import deformable_softmax_pool, deformable_sum_pool, peak_local_softmax


def smoothf1_loss(
        labels_pred: torch.Tensor,
        deltas_pred: torch.Tensor,
        deltas: torch.Tensor,
        p: torch.Tensor,
        max_distance: float = 3.0,
        kernel_size: Sequence[int] = (3, 3),
        temperature: float = 0.05,
        epsilon: float = 1e-7
) -> torch.Tensor:
    
    """Compute the SmoothF1 loss.

    Parameters
    ----------
    labels_pred : torch.Tensor
        Predicted labels.
    deltas_pred : torch.Tensor
        Predicted displacement vectors.
    deltas : torch.Tensor
        Ground truth displacement vectors.
    p : torch.Tensor
        Number of ground truth spots in each image.
    max_distance : float, optional
        Maximum distance for matching predicted and ground truth displacement vectors. Default is 3.
    temperature : float, optional
        Temperature parameter for softmax. Default is 0.05.
    kernel_size : Sequence[int], optional
        Kernel size of sum or max pooling operations. Default is (3, 3).
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.

    Returns
    -------
    smoothf1 : torch.Tensor
        SmoothF1 loss.
    """

    # Apply peak local softmax to the predicted labels.
    peaked_labels = peak_local_softmax(labels_pred, (3, 3), temperature)

    # Compute assignments and matches between predicted and ground truth displacement vectors.
    distances = torch.linalg.norm(deltas_pred - deltas, ord=2, dim=0)
    matches = torch.clamp(1 - distances / max_distance, min=0.0)
    assignments = deformable_sum_pool(peaked_labels, deltas, kernel_size).clamp(max=1.0)
    matches = deformable_softmax_pool(peaked_labels, deltas, matches, kernel_size, temperature)

    # Estimate the number of true positives, false positives, and false negatives.
    tp = torch.sum(assignments * matches)
    fp = torch.sum(peaked_labels) - tp
    fn = p - tp
    tp = tp + epsilon

    # Compute the SmoothF1 loss.
    smoothf1 = -2 * tp / (2 * tp + fp + fn)

    return smoothf1


def masked_l2_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        epsilon: float = 1e-7
) -> torch.Tensor:

    """Compute the L2 loss over masked pixels.

    Parameters
    ----------
    input : torch.Tensor
        Predicted values.
    target : torch.Tensor
        Ground truth values.
    mask : torch.Tensor
        Mask tensor where each pixel is a boolean for whether it should be included in the loss computation.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.

    Returns
    -------
    rmse : jax.Array
        Masked root-mean-square error.
    """

    l2 = (torch.linalg.norm(input - target, ord=2, dim=0) * mask).sum() / (mask.sum() + epsilon)

    return l2


def reduce_loss(
        loss: torch.Tensor,
        reduction: Optional[str] = 'mean'
) -> torch.Tensor:

    """Reduce the loss.

    Parameters
    ----------
    loss : torch.Tensor
        Loss tensor to be reduced.
    reduction : Optional[str], optional
        Loss reduction method. Supported methods are 'mean' and 'sum'. Default is 'mean'.

    Returns
    -------
    loss : torch.Tensor
        Reduced loss.

    Raises
    ------
    ValueError
        If the `reduction` method is not supported.
    """

    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise ValueError("Reduction method is not supported.")

    return loss


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

        # Build in_dims dynamically from the arguments.
        in_dims = tuple(axis if isinstance(arg, torch.Tensor) else None for arg in args)

        # Vectorize the loss function.
        loss = torch.vmap(loss_fn, in_dims=in_dims)(*args)

        # Reduce the loss.
        loss = reduce_loss(loss, reduction=reduction)

        return loss

    return wrapped_loss_fn


mean_smoothf1_loss = wrap_loss_fn(smoothf1_loss, axis=0, reduction='mean')
mean_masked_l2_loss = wrap_loss_fn(masked_l2_loss, axis=0, reduction='mean')
