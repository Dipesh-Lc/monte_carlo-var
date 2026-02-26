from __future__ import annotations

import numpy as np


def fit_garch11_arch(returns: np.ndarray, dist: str = "normal"):
    """
    Fit GARCH(1,1) using the `arch` package.

    returns: 1D array of returns (NOT losses)
    dist: "normal" or "t"
    """
    try:
        from arch import arch_model
    except ImportError as e:
        raise ImportError("Missing dependency: arch. Install with: pip install arch") from e

    r = np.asarray(returns, dtype=float).ravel()

    # arch likes % returns for numerical stability
    am = arch_model(r * 100.0, vol="Garch", p=1, q=1, mean="Constant", dist=dist)
    res = am.fit(disp="off")
    return res


def garch_forecast_sigma(res) -> float:
    """
    One-step-ahead sigma forecast in return units.
    """
    f = res.forecast(horizon=1, reindex=False)
    var = float(f.variance.values[-1, 0])
    sigma = float(np.sqrt(max(var, 1e-18)) / 100.0)
    return sigma


def garch_mean(res) -> float:
    """
    Constant mean estimate in return units.
    """
    mu = float(res.params.get("mu", 0.0)) / 100.0
    return mu


def garch_df_if_t(res) -> float | None:
    """
    Return df parameter if fitted with dist='t', else None.
    """
    # arch uses parameter name 'nu' for Student-t df
    if "nu" in res.params:
        return float(res.params["nu"])
    return None