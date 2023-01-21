import numpy as np
import pandas as pd

from collections.abc import Iterable
from functools import cached_property
from scipy import integrate, optimize, spatial


class SpotsMetrics:

    def __init__(self, y_true, y_pred):

        self.y_true = y_true
        self.y_pred = y_pred
        self.distance_matrices = {}

    def calculate(self, agg_metric='f1', distance_metric='euclidean', thresholds=np.linspace(0, 3, 50)):

        if not isinstance(thresholds, Iterable):
            thresholds = [thresholds]

        agg_list = []
        deltas_list = []
        offsets_list = []

        for threshold in thresholds:
            agg, deltas, offsets = self._calculate(agg_metric, distance_metric, threshold)
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

    def _calculate(self, agg_metric, distance_metric, threshold):

        tp, fp, fn, matches = self._get_probabilities(distance_metric, threshold)

        agg = None

        if agg_metric == 'f1':

            precision = tp / np.max((tp + fp, 1e-7))
            recall = tp / np.max((tp + fn, 1e-7))
            agg = 2 * precision * recall / np.max((precision + recall, 1e-7))

        elif agg_metric == 'precision':

            agg = tp / np.max((tp + fp, 1e-7))

        elif agg_metric == 'recall':

            agg = tp / np.max((tp + fn, 1e-7))

        deltas = self.y_true[matches[0]] - self.y_pred[matches[1]]
        offsets = self.distance_matrices[distance_metric][matches]

        return agg, deltas, offsets

    def _get_distance_matrix(self, distance_metric):

        if distance_metric in self.distance_matrices.keys():
            distance_matrix = self.distance_matrices[distance_metric]
        else:
            distance_matrix = spatial.distance.cdist(self.y_true, self.y_pred)
            self.distance_matrices[distance_metric] = distance_matrix

        return distance_matrix

    def _get_probabilities(self, distance_metric, threshold):

        distance_matrix = self._get_distance_matrix(distance_metric)

        y_true_matches = linear_sum_assignment(distance_matrix, threshold)
        y_pred_matches = linear_sum_assignment(distance_matrix.T, threshold)

        tp = len(y_true_matches[0])
        fp = len(self.y_pred) - len(y_pred_matches[0])
        fn = len(self.y_true) - len(y_true_matches[0])

        return tp, fp, fn, y_true_matches

    def _get_offsets(self, distance_metric, matches):

        offsets = self.distance_matrices[distance_metric][matches]

        return offsets


def linear_sum_assignment(cost_matrix, threshold):

    if cost_matrix.size == 0:

        matches = []

    else:

        if threshold is not None:
            cost_matrix = np.where(cost_matrix > threshold, cost_matrix.max(), cost_matrix)
        matches = optimize.linear_sum_assignment(cost_matrix)
        below_threshold = [i for i, match in enumerate(zip(*matches)) if cost_matrix[match] <= threshold]
        matches = (matches[0][below_threshold], matches[1][below_threshold])

    return matches


class SpotMetricsLegacy:

    def __init__(self, y_true, y_single_pred, id_pred, **kwargs):

        self.y_true = y_true
        self.y_single_pred = y_single_pred
        self.id_pred = id_pred

        self.smooth_l1_beta = 1

        allowed_keys = list(self.__dict__.keys())
        self.__dict__.update((key, value) for key, value in kwargs.items() if key in allowed_keys)
        rejected_keys = set(kwargs.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError(f"Invalid arguments in constructor: {rejected_keys}")

    @cached_property
    def l1(self):

        if len(self.y_true) > 0:

            _l1s = np.linalg.norm(self.y_true - self.y_single_pred, ord=1, axis=-1)
            _best_l1s_index = np.argmin(_l1s)
            _best_l1 = {
                'value': _l1s[_best_l1s_index],
                'match': _best_l1s_index
            }

        else:

            _best_l1 = {
                'value': np.inf,
                'match': None
            }

        return _best_l1

    @cached_property
    def l2(self):

        if len(self.y_true) > 0:

            _l2s = np.linalg.norm(self.y_true - self.y_single_pred, ord=2, axis=-1)
            _best_l2s_index = np.argmin(_l2s)
            _best_l2 = {
                'value': _l2s[_best_l2s_index],
                'match': _best_l2s_index
            }

        else:

            _best_l2 = {
                'value': np.inf,
                'match': None
            }

        return _best_l2

    @cached_property
    def smooth_l1(self):

        if len(self.y_true) > 0:

            diff = self.y_true - self.y_single_pred
            _l1s = np.linalg.norm(diff, ord=1, axis=-1)
            _l2s = np.linalg.norm(diff, ord=2, axis=-1)
            _criteria = _l1s < self.smooth_l1_beta

            _smooth_l1s = 0
            _smooth_l1s = _smooth_l1s + _criteria * 0.5 * _l2s / self.smooth_l1_beta
            _smooth_l1s = _smooth_l1s + (~_criteria) * (_l1s - 0.5 * self.smooth_l1_beta)

            _best_smooth_l1s_index = np.argmin(_smooth_l1s)
            _best_smooth_l1 = {
                'value': _smooth_l1s[_best_smooth_l1s_index],
                'match': _best_smooth_l1s_index
            }

        else:

            _best_smooth_l1 = {
                'value': np.inf,
                'match': None
            }

        return _best_smooth_l1


class SpotsMetricsLegacy:

    def __init__(self, y_true, y_pred, **kwargs):

        self.y_true = y_true
        self.y_pred = y_pred

        self._spot_metrics = [
            SpotMetricsLegacy(y_true, y_single_pred, i, **kwargs) for i, y_single_pred in enumerate(y_pred)
        ]

    def calculate(self, agg_metric, match_metric, threshold):

        matches = [
            spot_metrics for spot_metrics in self._spot_metrics
            if getattr(spot_metrics, match_metric)['value'] < threshold
        ]
        match_ids = np.unique([getattr(match, match_metric)['match'] for match in matches])

        tp = len(match_ids)
        fp = len(self.y_pred) - tp
        fn = len(self.y_true) - len(match_ids)

        if agg_metric == 'f1':

            precision = tp / np.max((tp + fp, 1e-07))
            recall = tp / np.max((tp + fn, 1e-07))
            f1 = 2 * precision * recall / np.max((precision + recall, 1e-07))

            return f1

        elif agg_metric == 'AP':

            ap = tp / (tp + fp + fn)

            return ap
