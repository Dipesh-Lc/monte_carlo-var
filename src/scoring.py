# src/scoring.py
from __future__ import annotations

import numpy as np


def fz0_score(losses: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> np.ndarray:
    """
    Fissler-Ziegel (FZ0) strictly consistent scoring function for (VaR, ES).
    Losses L_t, forecasts (VaR_t, ES_t), alpha in (0,1).

    One common form (for losses) is:
      S_t = (I{L>V}- (1-alpha)) * V/ES + L/ES + log(ES)

    Lower is better.

    Notes:
    - requires ES > 0.
    - we clamp ES away from 0 for numerical stability.
    """
    L = np.asarray(losses, dtype=float).ravel()
    V = np.asarray(var, dtype=float).ravel()
    E = np.asarray(es, dtype=float).ravel()
    if not (L.shape == V.shape == E.shape):
        raise ValueError("losses, var, es must have same shape")

    p = float(1.0 - alpha)
    E = np.maximum(E, 1e-12)

    I = (L > V).astype(float)
    score = (I - p) * (V / E) + (L / E) + np.log(E)
    return score


def mean_fz0(losses: np.ndarray, var: np.ndarray, es: np.ndarray, alpha: float) -> float:
    return float(np.mean(fz0_score(losses, var, es, alpha)))