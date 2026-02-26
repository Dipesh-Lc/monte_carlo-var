from __future__ import annotations

import numpy as np


def weighted_quantile(values: np.ndarray, quantile: float, sample_weight: np.ndarray) -> float:
    """
    Weighted quantile for values with nonnegative weights.
    Returns smallest x such that weighted CDF >= quantile.

    values: (n,)
    sample_weight: (n,)
    quantile: in (0,1)
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(sample_weight, dtype=float).ravel()

    if v.size != w.size:
        raise ValueError("values and sample_weight must have same length.")
    if v.size < 2:
        raise ValueError("Need at least 2 samples.")
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0,1).")
    if np.any(w < 0):
        raise ValueError("sample_weight must be nonnegative.")
    sw = w.sum()
    if sw <= 0:
        raise ValueError("sum of sample_weight must be > 0.")

    idx = np.argsort(v)
    v_sorted = v[idx]
    w_sorted = w[idx]

    cdf = np.cumsum(w_sorted) / sw
    j = np.searchsorted(cdf, quantile, side="left")
    j = min(max(int(j), 0), v_sorted.size - 1)
    return float(v_sorted[j])


def weighted_tail_mean(values: np.ndarray, threshold: float, sample_weight: np.ndarray) -> float:
    """
    Weighted mean of values where values >= threshold.
    """
    v = np.asarray(values, dtype=float).ravel()
    w = np.asarray(sample_weight, dtype=float).ravel()
    mask = v >= threshold
    if not np.any(mask):
        return float(threshold)
    ww = w[mask]
    vv = v[mask]
    s = ww.sum()
    if s <= 0:
        return float(threshold)
    return float(np.sum(ww * vv) / s)