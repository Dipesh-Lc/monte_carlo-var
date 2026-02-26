from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
from math import sqrt, log


def dkw_epsilon(n: int, delta: float) -> float:
    """
    DKW inequality epsilon such that:
      P( sup_x |F_n(x) - F(x)| > eps ) <= 2 exp(-2 n eps^2)
    so choose eps(n,delta)=sqrt((1/(2n)) * log(2/delta))
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")
    return sqrt((1.0 / (2.0 * n)) * log(2.0 / delta))


def dkw_quantile_bracket(
    samples: np.ndarray,
    alpha: float,
    delta: float = 0.05,
    losses: bool = True,
) -> Dict[str, Any]:
    """
    Finite-sample bracket for the alpha-quantile using DKW.

    With prob >= 1-delta:
      q_{alpha-eps} <= q_alpha <= q_{alpha+eps}
    where eps = dkw_epsilon(n, delta).

    We return empirical quantiles at (alpha-eps) and (alpha+eps), clipped to [0,1].
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1).")

    x = np.asarray(samples, dtype=float).ravel()
    if x.size < 10:
        raise ValueError("Need at least 10 samples.")
    L = -x if losses else x

    n = int(L.size)
    eps = dkw_epsilon(n, delta)

    lo_p = max(0.0, alpha - eps)
    hi_p = min(1.0, alpha + eps)

    q_lo = float(np.quantile(L, lo_p, method="linear"))
    q_hi = float(np.quantile(L, hi_p, method="linear"))

    return {
        "alpha": float(alpha),
        "delta": float(delta),
        "n": n,
        "eps": float(eps),
        "p_lo": float(lo_p),
        "p_hi": float(hi_p),
        "q_lo": float(q_lo),
        "q_hi": float(q_hi),
        "width": float(q_hi - q_lo),
    }