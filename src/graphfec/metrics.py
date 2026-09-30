"""Inspectable per-example metrics and paired, label-stratified intervals."""

import numpy as np


def stratified_interval(values, labels, resamples=10000, seed=2026):
    values, labels = np.asarray(values, dtype=float), np.asarray(labels)
    if len(values) != len(labels) or not len(values) or resamples < 1:
        raise ValueError("Bootstrap needs nonempty aligned observations and resamples >= 1")
    groups = [values[labels == label] for label in np.unique(labels)]
    rng = np.random.default_rng(seed)
    samples = np.zeros(resamples)
    # Chunk resamples to avoid a resamples-by-testset allocation.
    for start in range(0, resamples, 128):
        stop = min(start + 128, resamples)
        for group in groups:
            indices = rng.integers(len(group), size=(stop - start, len(group)))
            samples[start:stop] += group[indices].sum(1) / len(values)
    return np.quantile(samples, [0.025, 0.975]).tolist()
