import numpy as np
import pandas as pd

from scipy import integrate, optimize, spatial
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union


def compute_metrics(
        coords_pred: np.ndarray,
        coords: np.ndarray,
        evaluation_metrics: Union[str, Sequence[str]] = 'f1',
        distance_metric: str = 'euclidean',
        distance_thresholds: Union[float, Iterable] = 1.0,
        epsilon: float = 1e-7,
        return_df: bool = False
) -> Union[Dict[str, float], Tuple[Dict[str, float], pd.DataFrame]]:

    """Compute evaluation metrics from predicted and ground truth spot coordinates. Adapted from deepBlink.

    Parameters
    ----------
    coords_pred : np.ndarray
        Predicted spot coordinates.
    coords : np.ndarray
        Ground truth spot coordinates.
    evaluation_metrics : Union[str, Sequence[str]], optional
        Evaluation metric or a list of evaluation metrics. Supported evaluation metrics are 'f1', 'precision', 'recall',
        'tp', 'fp', and 'fn'. Default is 'f1'.
    distance_metric : str, optional
        Distance metric used to compute the distances between predicted and ground truth spot coordinates. Default is
        'euclidean'.
    distance_thresholds : Union[float, Iterable], optional
        Distance threshold or a list of distance thresholds for matching predicted and ground truth spot coordinates.
        Default is 1.0.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.
    return_df : bool, optional
        Whether to return a DataFrame with additional information. Default is False.

    Returns
    -------
    integral_metrics : Dict[str, float]
        Dictionary of aggregate metrics integrated over the distance thresholds.
    df : pd.DataFrame, optional
        DataFrame with additional information. Only returned if `return_df` is True.

    References
    ----------
    .. [1] Eichenberger, Bastian, et al. "BBQuercus/deepBlink: Threshold independent detection and localization of
           diffraction-limited spots." GitHub, https://github.com/BBQuercus/deepBlink.
    """

    # Compute the distance matrix between predicted and ground truth spot coordinates.
    distance_matrix = spatial.distance.cdist(coords, coords_pred, metric=distance_metric)

    # Convert `evaluation_metrics` to a tuple if necessary.
    if isinstance(evaluation_metrics, str):
        evaluation_metrics = (evaluation_metrics,)

    # Convert `distance_thresholds` to a tuple if necessary.
    if not isinstance(distance_thresholds, Iterable):
        distance_thresholds = (distance_thresholds, )

    # Create empty lists.
    metrics_list = []
    deltas_list = []
    offsets_list = []

    if return_df:

        # Loop over distance thresholds.
        for distance_threshold in distance_thresholds:

            # Compute evaluation metrics.
            metrics, coords_matches = _compute_metrics(distance_matrix, evaluation_metrics, distance_threshold, epsilon)

            # Compute deltas and offsets.
            deltas = coords[coords_matches[0]] - coords_pred[coords_matches[1]]
            offsets = distance_matrix[coords_matches]

            # Add evaluation metrics, deltas, and offsets.
            metrics_list.append(metrics)
            deltas_list.append(deltas)
            offsets_list.append(offsets)

    else:

        # Loop over distance thresholds.
        for distance_threshold in distance_thresholds:

            # Compute evaluation metrics.
            metrics, _ = _compute_metrics(distance_matrix, evaluation_metrics, distance_threshold, epsilon)

            # Add evaluation metrics.
            metrics_list.append(metrics)

    # Create an empty dictionary.
    integral_metrics = {}

    # Integrate evaluation metrics over distance thresholds.
    if len(distance_thresholds) > 1:
        for evaluation_metric in evaluation_metrics:
            integral_metrics[evaluation_metric] = (
                    integrate.trapz([metrics[evaluation_metric] for metrics in metrics_list], distance_thresholds) /
                    np.ptp(distance_thresholds))
    else:
        integral_metrics = metrics_list[0]

    if return_df:

        # Create a DataFrame with deltas, offsets, and evaluation metrics for each distance threshold.
        df = pd.DataFrame(
            {'thresholds': distance_thresholds, 'deltas': deltas_list, 'offsets': offsets_list} |
            {k: [metrics[k] for metrics in metrics_list] for k in evaluation_metrics}
        )

        return integral_metrics, df

    else:

        return integral_metrics


def _compute_metrics(
        distance_matrix: np.ndarray,
        evaluation_metrics: Union[str, Sequence[str]],
        distance_threshold: float,
        epsilon: float,
) -> Tuple[Dict[str, float], Tuple[np.ndarray, np.ndarray]]:

    """Compute evaluation metrics from a given distance matrix.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Distance matrix.
    evaluation_metrics : Union[str, Sequence[str]]
        Evaluation metric or a list of evaluation metrics. Supported evaluation metrics are 'f1', 'precision', 'recall',
        'tp', 'fp', and 'fn'.
    distance_threshold : float
        Distance threshold for matching predicted and ground truth spot coordinates.
    epsilon : float
        Small constant for numerical stability.
    """

    # Match predicted and ground truth spot coordinates.
    coords_matches = linear_sum_assignment(distance_matrix, distance_threshold)

    # Compute the number of true positives, false positives, and false negatives.
    tp = len(coords_matches[0])
    fp = distance_matrix.shape[1] - tp
    fn = distance_matrix.shape[0] - tp

    # Compute evaluation metrics.
    metrics = {}
    if 'f1' in evaluation_metrics:
        metrics['f1'] = 2 * tp / (2 * tp + fp + fn + epsilon)
    if 'precision' in evaluation_metrics:
        metrics['precision'] = tp / (tp + fp + epsilon)
    if 'recall' in evaluation_metrics:
        metrics['recall'] = tp / (tp + fn + epsilon)
    if 'tp' in evaluation_metrics:
        metrics['tp'] = tp
    if 'fp' in evaluation_metrics:
        metrics['fp'] = fp
    if 'fn' in evaluation_metrics:
        metrics['fn'] = fn

    return metrics, coords_matches


def linear_sum_assignment(
        cost_matrix: np.ndarray,
        threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:

    """Solve the linear sum assignment problem for a given cost matrix.

    Parameters
    ----------
    cost_matrix : np.ndarray
        Cost matrix.
    threshold : float, optional
        Threshold for matching. Default is None.

    Returns
    -------
    matches : Tuple[np.ndarray, np.ndarray]
        Tuple of row and column indices of the matches.
    """

    if threshold is not None:
        cost_matrix = np.where(cost_matrix > threshold, np.max(cost_matrix), cost_matrix)
    matches = optimize.linear_sum_assignment(cost_matrix)
    below_threshold = [i for i, match in enumerate(zip(*matches)) if cost_matrix[match] <= threshold]
    matches = (matches[0][below_threshold], matches[1][below_threshold])

    return matches
