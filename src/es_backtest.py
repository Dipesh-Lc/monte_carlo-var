# src/es_backtest.py
from __future__ import annotations

import numpy as np


def es_backtest_summary(losses: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> dict:
    """
    Simple ES diagnostics + exception series.

    Returns:
      - exception_rate, expected_rate
      - mean_exceed_loss, mean_es, es_ratio
      - n, n_exceed
      - exceptions (0/1 array)  [useful for Kupiec/Christoffersen]
    """
    L = np.asarray(losses, dtype=float).ravel()
    V = np.asarray(var, dtype=float).ravel()
    E = np.asarray(es, dtype=float).ravel()

    if not (L.shape == V.shape == E.shape):
        raise ValueError("losses, var, es must have same shape")

    exc = (L > V).astype(int)
    exc_rate = float(exc.mean())
    expected = float(1.0 - alpha)

    if exc.sum() == 0:
        mean_exceed = float("nan")
    else:
        mean_exceed = float(L[exc == 1].mean())

    mean_es = float(np.mean(E))
    es_ratio = float(mean_exceed / mean_es) if np.isfinite(mean_exceed) and mean_es > 0 else float("nan")

    return {
        "exception_rate": exc_rate,
        "expected_rate": expected,
        "mean_exceed_loss": mean_exceed,
        "mean_es": mean_es,
        "es_ratio": es_ratio,
        "n": int(L.size),
        "n_exceed": int(exc.sum()),
        "exceptions": exc,  # <-- used for UC/IND tests
    }