from __future__ import annotations

from typing import Dict, Any, Union, Sequence

import numpy as np
import pandas as pd

from src.simulate import _to_weights_array  # reuse your helper


ArrayLike = Union[np.ndarray, list]


def simulate_multivariate_t(
    mu: np.ndarray,
    cov: np.ndarray,
    nu: float,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw samples from multivariate Student-t with location mu, scale cov, df=nu.

    Representation:
      Z ~ N(0, cov)
      U ~ ChiSquare(nu)
      X = mu + Z / sqrt(U/nu)
    """
    if nu <= 2:
        # I *can* allow <=2, but variance becomes infinite.,
        # keep default nu > 2 for stability.
        raise ValueError("nu must be > 2 for finite covariance (recommended).")

    z = rng.multivariate_normal(mean=np.zeros_like(mu), cov=cov, size=int(n_sims))
    u = rng.chisquare(df=nu, size=int(n_sims))
    x = mu + z / np.sqrt(u[:, None] / nu)
    return x


def mc_multivariate_t(
    returns_df: pd.DataFrame,
    weights: Union[dict, ArrayLike],
    nu: float = 5.0,
    n_sims: int = 20000,
    seed: int = 0,
    ddof: int = 1,
) -> Dict[str, Any]:
    """
    Monte Carlo simulation of portfolio returns using multivariate Student-t scenarios.
    Parameters are estimated from historical returns:
      mu = sample mean, cov = sample covariance.
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
    scenarios = simulate_multivariate_t(mu=mu, cov=cov, nu=float(nu), n_sims=int(n_sims), rng=rng)
    port_returns = scenarios @ w

    return {
        "port_returns": port_returns.astype(float),
        "meta": {
            "method": "multivariate_t",
            "nu": float(nu),
            "n_sims": int(n_sims),
            "seed": int(seed),
            "ddof": int(ddof),
            "assets": cols,
            "mu": mu,
            "cov": cov,
            "weights": w,
        },
    }


def choose_nu_gridsearch(
    returns_df: pd.DataFrame,
    nu_grid: Sequence[float] = (3, 4, 5, 7, 10, 15, 30),
) -> float:
    """
    Simple heuristic: choose nu by matching empirical kurtosis (average across assets)
    to theoretical Student-t kurtosis: 3*(nu-2)/(nu-4) for nu>4.
    """
    X = returns_df.dropna().to_numpy(dtype=float)
    # sample excess kurtosis per asset (Fisher=False gives "plain kurtosis")
    # I'll compute manual kurtosis to avoid dependencies.
    mu = X.mean(axis=0)
    centered = X - mu
    m2 = np.mean(centered**2, axis=0)
    m4 = np.mean(centered**4, axis=0)
    kurt = m4 / (m2**2 + 1e-12)  # plain kurtosis
    target = float(np.mean(kurt))

    best_nu = None
    best_err = float("inf")
    for nu in nu_grid:
        if nu <= 4:
            continue  # kurtosis infinite
        theo = 3.0 * (nu - 2.0) / (nu - 4.0)
        err = abs(theo - target)
        if err < best_err:
            best_err = err
            best_nu = nu

    # fallback
    return float(best_nu) if best_nu is not None else 10.0