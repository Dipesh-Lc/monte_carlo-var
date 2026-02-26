from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, list]


def _to_weights_array(weights: Union[dict, ArrayLike], columns: list[str]) -> np.ndarray:
    """
    Convert weights (dict keyed by asset or array-like) into a numpy array aligned to columns.
    """
    if isinstance(weights, dict):
        w = np.array([weights[c] for c in columns], dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ValueError("weights must be a 1D array-like or a dict keyed by asset names.")
        if len(w) != len(columns):
            raise ValueError(f"weights length {len(w)} must match number of assets {len(columns)}.")
    # Normalize if slightly off
    s = w.sum()
    if not np.isclose(s, 1.0):
        if s == 0:
            raise ValueError("weights sum to 0; cannot normalize.")
        w = w / s
    return w


def mc_multivariate_normal(
    returns_df: pd.DataFrame,
    weights: Union[dict, ArrayLike],
    n_sims: int = 20000,
    seed: int = 0,
    ddof: int = 1,
) -> Dict[str, Any]:
    """
    Monte Carlo simulation of portfolio returns using multivariate normal scenarios.

    returns_df: DataFrame of historical returns with columns as assets.
    weights: dict keyed by columns or array-like aligned to columns.
    n_sims: number of simulations.
    seed: RNG seed.
    ddof: degrees of freedom for covariance (1 = sample covariance).
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame of returns.")
    if returns_df.shape[0] < 5:
        raise ValueError("returns_df must have at least 5 rows of returns.")
    if returns_df.isna().any().any():
        returns_df = returns_df.dropna()

    cols = list(returns_df.columns)
    w = _to_weights_array(weights, cols)

    mu = returns_df.mean(axis=0).to_numpy(dtype=float)
    cov = returns_df.cov(ddof=ddof).to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    scenarios = rng.multivariate_normal(mean=mu, cov=cov, size=int(n_sims))

    port_returns = scenarios @ w

    return {
        "port_returns": port_returns.astype(float),
        "meta": {
            "method": "multivariate_normal",
            "n_sims": int(n_sims),
            "seed": int(seed),
            "ddof": int(ddof),
            "assets": cols,
            "mu": mu,
            "cov": cov,
            "weights": w,
        },
    }


def mc_bootstrap(
    returns_df: pd.DataFrame,
    weights: Union[dict, ArrayLike],
    n_sims: int = 20000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Nonparametric bootstrap Monte Carlo of portfolio returns:
    sample historical return rows with replacement.
    """
    if not isinstance(returns_df, pd.DataFrame):
        raise TypeError("returns_df must be a pandas DataFrame of returns.")
    if returns_df.shape[0] < 5:
        raise ValueError("returns_df must have at least 5 rows of returns.")
    if returns_df.isna().any().any():
        returns_df = returns_df.dropna()

    cols = list(returns_df.columns)
    w = _to_weights_array(weights, cols)

    X = returns_df.to_numpy(dtype=float)
    n = X.shape[0]

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=int(n_sims))
    scenarios = X[idx, :]

    port_returns = scenarios @ w

    return {
        "port_returns": port_returns.astype(float),
        "meta": {
            "method": "bootstrap",
            "n_sims": int(n_sims),
            "seed": int(seed),
            "assets": cols,
            "weights": w,
            "n_hist": int(n),
        },
    }