"""Exact nearest-neighbor metrics and fixed-direction sliced Wasserstein-1."""

import numpy as np
from scipy.spatial import cKDTree

from .data import ConditionScheme


def points(value, name="points"):
    value = np.asarray(value, dtype=np.float64)
    #region Input validation
    if value.ndim != 2 or value.shape[1] != 2 or len(value) == 0:
        raise ValueError(f"{name} must have nonempty shape [N, 2].")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite.")
    #endregion
    return value


def endpoint_rmse(samples, reference):
    samples, reference = points(samples), points(reference, "reference")
    #region Input validation
    if samples.shape != reference.shape:
        raise ValueError("Samples and reference must have identical shapes.")
    #endregion
    return float(np.sqrt(np.square(samples - reference).sum(axis=1).mean()))


class GeometryMetrics:
    """Cache the target KD-tree and sorted projections across condition counts."""

    def __init__(self, targets, num_directions=256, direction_seed=1729):
        self.targets = points(targets, "targets")
        #region Configuration validation
        if type(num_directions) is not int or num_directions < 1:
            raise ValueError("num_directions must be a positive integer.")
        #endregion
        angles = np.random.default_rng(direction_seed).uniform(0, 2 * np.pi, num_directions)
        self.directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
        self.tree = cKDTree(self.targets)
        self.sorted_targets = np.empty((len(self.targets), num_directions), dtype=np.float64)
        for start in range(0, num_directions, 16):
            stop = start + 16
            self.sorted_targets[:, start:stop] = np.sort(
                self.targets @ self.directions[start:stop].T, axis=0,
            )

    def conditional_distances(self, samples, group_ids, scheme: ConditionScheme):
        group_ids = np.asarray(group_ids)
        #region Input validation
        if scheme.num_targets != len(self.targets):
            raise ValueError("Condition scheme and reference dataset sizes differ.")
        if group_ids.shape != (len(samples),) or group_ids.dtype.kind not in "iu":
            raise ValueError("group_ids must contain one integer per sample.")
        if np.any((group_ids < 0) | (group_ids >= scheme.num_conditions)):
            raise ValueError("Condition group outside the dataset.")
        #endregion
        size = scheme.targets_per_condition
        if size == 1:
            # One target per condition: direct distances avoid building N tiny trees.
            return np.linalg.norm(samples - self.targets[group_ids], axis=1)
        if scheme.num_conditions == 1:
            return self.tree.query(samples)[0]
        distances = np.empty(len(samples))
        order = np.argsort(group_ids, kind="stable")
        # Sort once so each group can be queried together, including unordered inputs.
        groups, counts = np.unique(group_ids[order], return_counts=True)
        offset = 0
        for group, count in zip(groups, counts):
            selection = order[offset:offset + count]
            tree = cKDTree(self.targets[group * size:(group + 1) * size])
            distances[selection] = tree.query(samples[selection])[0]
            offset += count
        return distances

    def evaluate(self, samples, group_ids, scheme, thresholds):
        samples = points(samples, "samples")
        #region Input validation
        if len(samples) != len(self.targets):
            raise ValueError("Sliced Wasserstein comparison requires N generated samples.")
        if not thresholds or any(not np.isfinite(e) or e <= 0 for e in thresholds):
            raise ValueError("Distance thresholds must be finite and positive.")
        #endregion
        precision_distance = self.tree.query(samples)[0]
        recall_distance = cKDTree(samples).query(self.targets)[0]
        conditional_distance = self.conditional_distances(samples, group_ids, scheme)
        projected_error = 0.0
        # Equal-size 1D Wasserstein-1 is the mean absolute difference of sorted points.
        # Process directions in chunks to limit temporary projection memory.
        for start in range(0, len(self.directions), 16):
            stop = start + 16
            projected = np.sort(samples @ self.directions[start:stop].T, axis=0)
            projected_error += np.abs(projected - self.sorted_targets[:, start:stop]).sum()
        return {
            "sliced_wasserstein_1": float(projected_error / self.sorted_targets.size),
            "thresholds": [{
                "epsilon": epsilon,
                "geometric_precision": float((precision_distance <= epsilon).mean()),
                "geometric_recall": float((recall_distance <= epsilon).mean()),
                "conditional_precision": float((conditional_distance <= epsilon).mean()),
            } for epsilon in thresholds],
        }
