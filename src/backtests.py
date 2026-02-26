# src/backtests.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.simulate import mc_multivariate_normal, mc_bootstrap
from src.risk import var_cvar_from_returns


# =============================================================================
# Statistical backtests (Kupiec UC + Christoffersen IND)
# =============================================================================

def kupiec_pof_test(n_exceptions: int, n_obs: int, alpha: float = 0.99) -> tuple[float, float]:
    """
    Kupiec (1995) Proportion of Failures (POF) / Unconditional Coverage (UC) test.

    H0: exception probability == p where p = 1 - alpha

    Returns:
      (LR_stat, p_value)  where LR_stat ~ ChiSq(1) asymptotically.
    """
    p = float(1.0 - alpha)
    x = int(n_exceptions)
    T = int(n_obs)

    if T <= 0:
        raise ValueError("n_obs must be > 0")
    if x < 0 or x > T:
        raise ValueError("n_exceptions must be between 0 and n_obs")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    # Boundary cases (avoid log(0))
    if x == 0:
        # LR = -2[ log L(p) - log L(phat=0) ]
        # L(phat=0) -> (1-0)^T = 1 so log L = 0
        lr = -2.0 * (T * np.log(1.0 - p))
        return float(lr), float(1.0 - chi2.cdf(lr, df=1))

    if x == T:
        # L(phat=1) = 1 so log L = 0
        lr = -2.0 * (T * np.log(p))
        return float(lr), float(1.0 - chi2.cdf(lr, df=1))

    phat = x / T
    lr = -2.0 * (
        (T - x) * np.log((1.0 - p) / (1.0 - phat)) +
        x * np.log(p / phat)
    )
    pval = float(1.0 - chi2.cdf(lr, df=1))
    return float(lr), float(pval)


def kupiec_uc_from_exceptions(exceptions: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """
    Convenience wrapper: Kupiec UC test from a 0/1 exception series.

    Returns (LR_stat, p_value).
    """
    exc = np.asarray(exceptions, dtype=int).ravel()
    n_obs = int(exc.size)
    n_exc = int(exc.sum())
    return kupiec_pof_test(n_exc, n_obs, alpha=alpha)


def christoffersen_ind_test(exceptions: np.ndarray) -> tuple[float, float]:
    """
    Christoffersen (1998) independence test for exception clustering.

    H0: exceptions are independent over time (iid Bernoulli)
    H1: exceptions follow a 2-state Markov chain

    Returns:
      (LR_ind, p_value) where LR_ind ~ ChiSq(1) asymptotically.
    """
    x = np.asarray(exceptions, dtype=int).ravel()
    if x.size < 2:
        return float("nan"), float("nan")

    x0 = x[:-1]
    x1 = x[1:]

    n00 = int(np.sum((x0 == 0) & (x1 == 0)))
    n01 = int(np.sum((x0 == 0) & (x1 == 1)))
    n10 = int(np.sum((x0 == 1) & (x1 == 0)))
    n11 = int(np.sum((x0 == 1) & (x1 == 1)))

    # Transition probabilities
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    # Unconditional probability of exception
    total = n00 + n01 + n10 + n11
    if total <= 0:
        return float("nan"), float("nan")
    pi = (n01 + n11) / total

    # Safe log-likelihood for Bernoulli counts
    def ll(n0: int, n1: int, p: float) -> float:
        p = float(np.clip(p, 1e-15, 1.0 - 1e-15))
        return n1 * np.log(p) + n0 * np.log(1.0 - p)

    # Null: iid Bernoulli(pi)
    ll0 = ll(n00 + n10, n01 + n11, pi)

    # Alternative: Markov chain
    ll1 = ll(n00, n01, pi01) + ll(n10, n11, pi11)

    lr_ind = float(-2.0 * (ll0 - ll1))
    pval = float(1.0 - chi2.cdf(lr_ind, df=1))
    return lr_ind, pval


# =============================================================================
# Rolling VaR backtest engine (optional utility)
# =============================================================================

def rolling_backtest(
    returns_df: pd.DataFrame,
    weights,
    method: str = "mc_normal",
    window: int = 500,
    n_sims: int = 30_000,
    alpha: float = 0.99,
    seed: int = 42,
):
    """
    Rolling one-step-ahead 1-day VaR backtest.

    method:
      - "parametric"
      - "mc_normal"
      - "mc_bootstrap"

    Returns:
      results_df with columns: VaR, RealizedLoss, Exception
      stats dict with Kupiec UC test
    """
    rets = returns_df.copy()

    w = np.asarray(weights, dtype=float).ravel()
    if w.size != rets.shape[1]:
        raise ValueError("weights must align with returns_df columns")
    w = w / w.sum()

    # realized portfolio return series
    port_realized = rets.values @ w
    realized_losses = -port_realized

    var_list, loss_list, exc_list = [], [], []

    for t in range(window, len(rets)):
        train = rets.iloc[t - window:t]

        if method == "parametric":
            from src.risk import parametric_var
            var_t = parametric_var(train, w, alpha=alpha)

        elif method == "mc_normal":
            sims = mc_multivariate_normal(train, w, n_sims=n_sims, seed=seed + t)
            stats = var_cvar_from_returns(sims["port_returns"], alpha=alpha)
            var_t = stats["VaR"]

        elif method == "mc_bootstrap":
            sims = mc_bootstrap(train, w, n_sims=n_sims, seed=seed + t)
            stats = var_cvar_from_returns(sims["port_returns"], alpha=alpha)
            var_t = stats["VaR"]

        else:
            raise ValueError("method must be parametric, mc_normal, or mc_bootstrap")

        loss_t = float(realized_losses[t])
        exc_t = int(loss_t > var_t)

        var_list.append(float(var_t))
        loss_list.append(loss_t)
        exc_list.append(exc_t)

    idx = rets.index[window:]
    out = pd.DataFrame({"VaR": var_list, "RealizedLoss": loss_list, "Exception": exc_list}, index=idx)

    n_exc = int(out["Exception"].sum())
    n_obs = int(len(out))
    exc_rate = n_exc / n_obs if n_obs else np.nan
    lr, pval = kupiec_pof_test(n_exc, n_obs, alpha=alpha)

    stats = {
        "method": method,
        "alpha": float(alpha),
        "window": int(window),
        "n_obs": int(n_obs),
        "n_exceptions": int(n_exc),
        "exception_rate": float(exc_rate),
        "kupiec_LR": float(lr),
        "kupiec_p_value": float(pval),
    }
    return out, stats