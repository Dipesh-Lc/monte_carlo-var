# src/risk.py
from __future__ import annotations

from typing import Dict, Any
import numpy as np


def var_cvar_from_returns(
    port_returns: np.ndarray,
    alpha: float = 0.99,
    losses: bool = True,
) -> Dict[str, Any]:
    """
    Compute empirical VaR and CVaR from portfolio returns.

    If losses=True, define losses L = -R.
    VaR_alpha = quantile_alpha(L)  (e.g., 99% quantile of loss)
    CVaR_alpha = E[L | L >= VaR_alpha]
    """
    x = np.asarray(port_returns, dtype=float).reshape(-1)
    if x.size < 10:
        raise ValueError("port_returns must have at least 10 samples.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    losses_arr = -x if losses else x

    # Empirical quantile (VaR)
    var = float(np.quantile(losses_arr, alpha, method="linear"))

    # Tail average (CVaR)
    tail = losses_arr[losses_arr >= var]
    cvar = float(np.mean(tail)) if tail.size > 0 else var

    return {"VaR": var, "CVaR": cvar, "alpha": float(alpha), "n": int(x.size)}


def parametric_var(returns_df, weights, alpha=0.99):
    """
    Parametric (Gaussian) VaR for portfolio loss L=-R using sample mean/cov.

    returns_df: pandas DataFrame (asset returns)
    weights: array aligned with columns
    """
    import numpy as np

    X = returns_df.to_numpy(dtype=float)
    w = np.asarray(weights, dtype=float).ravel()
    w = w / w.sum()

    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False, ddof=1)

    mean_p = float(mu @ w)
    var_p = float(w @ cov @ w)
    std_p = float(np.sqrt(max(var_p, 1e-18)))

    # VaR for loss at alpha: VaR(L)= -q_{1-alpha}(R)
    from src.theory import inv_norm_cdf
    z = inv_norm_cdf(1.0 - alpha)  # negative
    qR = mean_p + z * std_p
    return float(-qR)