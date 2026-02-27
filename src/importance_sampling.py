from __future__ import annotations

from typing import Dict, Any, Union

import numpy as np
import pandas as pd

from src.simulate import _to_weights_array
from src.weighted_stats import weighted_quantile, weighted_tail_mean


ArrayLike = Union[np.ndarray, list]


def _estimate_params(returns_df: pd.DataFrame, ddof: int = 1):
    mu = returns_df.mean(axis=0).to_numpy(dtype=float)
    cov = returns_df.cov(ddof=ddof).to_numpy(dtype=float)
    return mu, cov


def _log_lr_gaussian(
    x: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    cov_inv: np.ndarray,
) -> np.ndarray:
    """
    log likelihood ratio log(f0/f1) where
      f0 = N(mu0, Sigma), f1 = N(mu1, Sigma)

    log(f0/f1) = -0.5[(x-mu0)^T S^{-1}(x-mu0) - (x-mu1)^T S^{-1}(x-mu1)]
    """
    d0 = x - mu0
    d1 = x - mu1
    q0 = np.einsum("ij,jk,ik->i", d0, cov_inv, d0)
    q1 = np.einsum("ij,jk,ik->i", d1, cov_inv, d1)
    return -0.5 * (q0 - q1)


def mc_is_normal_shift(
    returns_df: pd.DataFrame,
    weights: Union[dict, ArrayLike],
    n_sims: int = 20000,
    seed: int = 0,
    ddof: int = 1,
    shift_scale: float = 1.0,
) -> Dict[str, Any]:
    """
    Importance sampling for tail VaR/CVaR under a fitted multivariate normal model.

    I will fit mu, cov from historical returns, then sample from a shifted-mean proposal:
        X ~ N(mu + theta, cov)

    theta is chosen along the *portfolio loss direction* in a calibrated way:
      - portfolio return: Rp = w^T X
      - portfolio loss:   L  = -Rp
      - desired mean shift in portfolio return: w^T theta = -k * sigma_p
        where sigma_p = sqrt(w^T cov w), k = shift_scale

    A stable choice is:
      v = cov w
      theta = -(k * sigma_p) * v / (w^T v)
    which guarantees w^T theta = -k*sigma_p.

    I will compute likelihood ratio weights w_is = f0(x)/f1(x) and return:
      - port_returns, losses, is_weights, ESS diagnostic
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame of returns.")
    if returns_df.isna().any().any():
        returns_df = returns_df.dropna()

    cols = list(returns_df.columns)
    w = _to_weights_array(weights, cols)

    mu, cov = _estimate_params(returns_df, ddof=ddof)
    cov_inv = np.linalg.inv(cov)

    # Portfolio volatility under fitted normal
    sigma_p2 = float(w @ cov @ w)
    sigma_p = float(np.sqrt(max(sigma_p2, 1e-18)))

    # Stable calibrated shift along portfolio direction
    v = cov @ w
    denom = float(w @ v)  # == sigma_p^2
    theta = -(float(shift_scale) * sigma_p) * (v / (denom + 1e-18))
    mu_prop = mu + theta

    rng = np.random.default_rng(seed)
    x_prop = rng.multivariate_normal(mean=mu_prop, cov=cov, size=int(n_sims))

    # Importance weights: w(x) = f0(x)/f1(x)
    logw = _log_lr_gaussian(x_prop, mu0=mu, mu1=mu_prop, cov_inv=cov_inv)

    # stabilize (avoid overflow/underflow)
    logw = logw - np.max(logw)
    w_is = np.exp(logw)

    # Effective sample size diagnostic
    w_norm = w_is / (w_is.sum() + 1e-300)
    ess = float(1.0 / np.sum(w_norm**2))

    port_returns = x_prop @ w
    losses = -port_returns

    return {
        "port_returns": port_returns.astype(float),
        "losses": losses.astype(float),
        "is_weights": w_is.astype(float),
        "meta": {
            "method": "is_normal_shift",
            "n_sims": int(n_sims),
            "seed": int(seed),
            "ddof": int(ddof),
            "shift_scale": float(shift_scale),
            "sigma_p": float(sigma_p),
            "theta_port_shift": float(w @ theta),  # should be approx -shift_scale*sigma_p
            "ess": ess,
            "assets": cols,
        },
    }


def is_var_cvar(losses: np.ndarray, weights: np.ndarray, alpha: float) -> Dict[str, float]:
    """
    Weighted VaR/CVaR from importance sampling sample.
    """
    losses = np.asarray(losses, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    var = weighted_quantile(losses, alpha, w)
    cvar = weighted_tail_mean(losses, var, w)
    return {"VaR": float(var), "CVaR": float(cvar)}