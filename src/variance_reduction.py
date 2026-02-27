from __future__ import annotations

from typing import Dict, Any, Union
import numpy as np
import pandas as pd

from src.simulate import _to_weights_array


ArrayLike = Union[np.ndarray, list]


def _estimate_params(returns_df: pd.DataFrame, ddof: int = 1):
    mu = returns_df.mean(axis=0).to_numpy(dtype=float)
    cov = returns_df.cov(ddof=ddof).to_numpy(dtype=float)
    return mu, cov


def simulate_normal_plain(
    returns_df: pd.DataFrame,
    weights: Union[dict, ArrayLike],
    n_sims: int,
    seed: int,
    ddof: int = 1,
) -> np.ndarray:
    cols = list(returns_df.columns)
    w = _to_weights_array(weights, cols)

    mu, cov = _estimate_params(returns_df, ddof=ddof)
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(mean=mu, cov=cov, size=int(n_sims))
    return (X @ w).astype(float)


def simulate_normal_antithetic(
    returns_df: pd.DataFrame,
    weights: Union[dict, ArrayLike],
    n_sims: int,
    seed: int,
    ddof: int = 1,
) -> np.ndarray:
    """
    Antithetic sampling: generate Z and -Z in the MVN space.
    We simulate half scenarios and mirror them.
    """
    cols = list(returns_df.columns)
    w = _to_weights_array(weights, cols)

    mu, cov = _estimate_params(returns_df, ddof=ddof)
    rng = np.random.default_rng(seed)

    m = int(n_sims // 2)
    X1 = rng.multivariate_normal(mean=np.zeros_like(mu), cov=cov, size=m)
    X2 = -X1

    # Add mean
    S = np.vstack([mu + X1, mu + X2])
    if S.shape[0] < n_sims:
        # one extra draw if odd
        extra = rng.multivariate_normal(mean=mu, cov=cov, size=1)
        S = np.vstack([S, extra])

    return (S[:n_sims] @ w).astype(float)


def var_estimator(losses: np.ndarray, alpha: float) -> float:
    return float(np.quantile(losses, alpha, method="linear"))


def cvar_estimator(losses: np.ndarray, alpha: float) -> float:
    v = var_estimator(losses, alpha)
    tail = losses[losses >= v]
    return float(np.mean(tail)) if tail.size else v


def control_variate_adjusted_losses(
    port_returns: np.ndarray,
    alpha: float,
    control: str = "loss",
) -> np.ndarray:
    """
    Control variate idea:
    Use a control variable with known expectation to reduce variance.
    A simple choice is the *loss* itself with known mean under the fitted normal model,
    but since I am working with simulated draws from that normal, I can use sample mean
    (or set target=0 if returns are demeaned).

    Practical choice:
    - control = "centered_loss": use (L - mean(L)) as control with target 0.
    This doesn't change VaR bias much, but can reduce noise in tail estimates.
    """
    R = np.asarray(port_returns, dtype=float).ravel()
    L = -R
    if control == "centered_loss":
        C = L - np.mean(L)
        # optimal beta for variance reduction in estimating a mean would be Cov/Var
        # For quantiles this is heuristic; still can stabilize empirical tail.
        beta = np.cov(L, C, ddof=1)[0, 1] / (np.var(C, ddof=1) + 1e-12)
        L_adj = L - beta * C  # target E[C]=0
        return L_adj
    else:
        raise ValueError("control must be 'centered_loss'.")