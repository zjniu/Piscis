import numpy as np
import pandas as pd

from collections.abc import Iterable
from scipy import integrate, optimize, spatial


def calculate_metric(y_true, y_pred, agg_metric='f1', distance_metric='euclidean', thresholds=1, extended=False):

    distance_matrix = spatial.distance.cdist(y_true, y_pred, metric=distance_metric)

    if not isinstance(thresholds, Iterable):
        thresholds = [thresholds]

    if extended:

        agg_list = []
        deltas_list = []
        offsets_list = []

        for threshold in thresholds:
            agg, deltas, offsets = \
                _calculate_metric(y_true, y_pred, agg_metric, distance_matrix, threshold, extended)
            agg_list.append(agg)
            deltas_list.append(deltas)
            offsets_list.append(offsets)

        if len(thresholds) > 1:
            agg = integrate.trapz(agg_list, thresholds) / np.ptp(thresholds)
        else:
            agg = agg_list[0]
        flattened_offsets_list = [offset for offsets in offsets_list for offset in offsets]
        if len(flattened_offsets_list) > 0:
            offset = np.mean([offset for offsets in offsets_list for offset in offsets])
        else:
            offset = 0.0
        deltas_list = [np.array(deltas) for deltas in deltas_list]
        offset_list = [np.mean(offsets) if len(offsets) > 0 else 0.0 for offsets in offsets_list]

        df = pd.DataFrame(
            {
                "thresholds": thresholds,
                "agg": agg_list,
                "deltas": deltas_list,
                "offset": offset_list,
            }
        )

        return agg, offset, df

    else:

        agg_list = [_calculate_metric(y_true, y_pred, agg_metric, distance_matrix, threshold, extended)
                    for threshold in thresholds]
        if len(thresholds) > 1:
            agg = integrate.trapz(agg_list, thresholds) / np.ptp(thresholds)
        else:
            agg = agg_list[0]

        return agg


def _calculate_metric(y_true, y_pred, agg_metric, distance_matrix, threshold, extended):

    y_true_matches = linear_sum_assignment(distance_matrix, threshold)
    y_pred_matches = linear_sum_assignment(distance_matrix.T, threshold)

    tp = len(y_true_matches[0])
    fp = len(y_pred) - len(y_pred_matches[0])
    fn = len(y_true) - len(y_true_matches[0])

    agg = None

    if agg_metric == 'f1':

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        agg = 2 * precision * recall / (precision + recall + 1e-7)

    elif agg_metric == 'precision':

        agg = tp / (tp + fp + 1e-7)

    elif agg_metric == 'recall':

        agg = tp / (tp + fn + 1e-7)

    if extended:

        deltas = y_true[y_true_matches[0]] - y_pred[y_true_matches[1]]
        offsets = distance_matrix[y_true_matches]

        return agg, deltas, offsets

    else:

        return agg


def linear_sum_assignment(cost_matrix, threshold):

    if threshold is not None:
        cost_matrix = np.where(cost_matrix > threshold, cost_matrix.max(), cost_matrix)
    matches = optimize.linear_sum_assignment(cost_matrix)
    below_threshold = [i for i, match in enumerate(zip(*matches)) if cost_matrix[match] <= threshold]
    matches = (matches[0][below_threshold], matches[1][below_threshold])

    return matches
