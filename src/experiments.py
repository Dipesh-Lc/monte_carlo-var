# src/experiments.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from src.scoring import mean_fz0
from src.simulate import mc_multivariate_normal, mc_bootstrap
from src.risk import var_cvar_from_returns
from src.heavy_tail import mc_multivariate_t, choose_nu_gridsearch, simulate_multivariate_t
from src.theory import quantile_asymptotic_ci, inv_norm_cdf, t_quantile
from src.bounds import dkw_quantile_bracket
from src.variance_reduction import simulate_normal_plain, simulate_normal_antithetic, var_estimator, cvar_estimator
from src.importance_sampling import mc_is_normal_shift, is_var_cvar
from src.es_backtest import es_backtest_summary
from src.garch_model import fit_garch11_arch, garch_forecast_sigma, garch_mean, garch_df_if_t
from src.backtests import kupiec_uc_from_exceptions, christoffersen_ind_test


def _parse_weights(weights_str: str | None, weights_file: str | None) -> Union[dict, list]:
    """
    Parse weights from:
    - --weights-file path to JSON file (dict or list)
    """
    if weights_file:
        p = Path(weights_file)
        if not p.exists():
            raise FileNotFoundError(f"weights file not found: {weights_file}")
        txt = p.read_text(encoding="utf-8-sig").strip()
        if not txt:
            raise ValueError(f"weights file is empty: {weights_file}")
        w = json.loads(txt)
        if not isinstance(w, (dict, list)):
            raise ValueError("weights file must contain a JSON dict or list.")
        return w

    if not weights_str:
        raise ValueError("Provide --weights or --weights-file")

    s = weights_str.strip()

    # Try strict JSON first
    try:
        w = json.loads(s)
    except json.JSONDecodeError:
        # Fallback: allow Python dict style with single quotes
        try:
            s2 = s.replace("'", '"')
            w = json.loads(s2)
        except json.JSONDecodeError as e:
            raise ValueError(
                "weights must be JSON. Examples:\n"
                '  --weights "{\\"AAPL\\":0.5,\\"MSFT\\":0.5}"\n'
                '  --weights "[0.5,0.5]"\n'
                "  --weights-file weights.json"
            ) from e

    if not isinstance(w, (dict, list)):
        raise ValueError("weights must be a JSON dict or JSON list.")
    return w


def _load_returns_csv(path: str) -> pd.DataFrame:
    """
    Load returns CSV.

    Expected:
    - columns = asset names (numeric)
    - rows = time points
    - values = returns (simple returns or log returns)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Returns file not found: {path}")

    df = pd.read_csv(p)

    # Try date index
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.set_index("Date")
        except Exception:
            pass
    else:
        # Try first 1-2 columns for date-like values
        for c in df.columns[:2]:
            if "date" in c.lower() or "time" in c.lower():
                try:
                    df[c] = pd.to_datetime(df[c])
                    df = df.set_index(c)
                except Exception:
                    pass
                break

    # Keep only numeric columns (assets)
    num_df = df.select_dtypes(include=[np.number]).copy()
    if num_df.shape[1] < 1:
        raise ValueError("No numeric asset return columns found in returns CSV.")
    num_df = num_df.dropna()
    if num_df.shape[0] < 5:
        raise ValueError("Returns CSV must contain at least 5 rows after dropping NaNs.")
    return num_df


def exp01_classical_var(
    returns_path: str,
    weights: Union[dict, list],
    n_sims: int = 20000,
    alpha: float = 0.99,
    seed: int = 0,
    out_csv: str = "paper/tables/exp01_classical_var.csv",
) -> pd.DataFrame:
    """
    Experiment 01:
    - simulate portfolio returns using multivariate normal and bootstrap
    - compute VaR & CVaR (loss-based)
    - save a results table
    """
    returns_df = _load_returns_csv(returns_path)

    sim_normal = mc_multivariate_normal(returns_df, weights, n_sims=n_sims, seed=seed)
    sim_boot = mc_bootstrap(returns_df, weights, n_sims=n_sims, seed=seed)

    risk_normal = var_cvar_from_returns(sim_normal["port_returns"], alpha=alpha, losses=True)
    risk_boot = var_cvar_from_returns(sim_boot["port_returns"], alpha=alpha, losses=True)

    rows = [
        {
            "method": "normal_mc",
            "alpha": alpha,
            "VaR": risk_normal["VaR"],
            "CVaR": risk_normal["CVaR"],
            "n_sims": n_sims,
            "seed": seed,
            "n_hist": int(returns_df.shape[0]),
            "nu": np.nan,
        },
        {
            "method": "bootstrap_mc",
            "alpha": alpha,
            "VaR": risk_boot["VaR"],
            "CVaR": risk_boot["CVaR"],
            "n_sims": n_sims,
            "seed": seed,
            "n_hist": int(returns_df.shape[0]),
            "nu": np.nan,
        },
    ]

    res = pd.DataFrame(rows)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)

    return res


def exp02_normal_vs_t(
    returns_path: str,
    weights: Union[dict, list],
    n_sims: int = 20000,
    alpha: float = 0.99,
    seed: int = 0,
    nu: float | None = None,
    out_csv: str = "paper/tables/exp02_normal_vs_t.csv",
) -> pd.DataFrame:
    """
    Experiment 02:
    Compare VaR/CVaR under:
      - multivariate normal MC
      - bootstrap MC
      - multivariate Student-t MC (nu fixed or chosen by grid search)
    """
    returns_df = _load_returns_csv(returns_path)

    if nu is None:
        nu = choose_nu_gridsearch(returns_df)

    sim_normal = mc_multivariate_normal(returns_df, weights, n_sims=n_sims, seed=seed)
    sim_boot = mc_bootstrap(returns_df, weights, n_sims=n_sims, seed=seed)
    sim_t = mc_multivariate_t(returns_df, weights, nu=nu, n_sims=n_sims, seed=seed)

    risk_normal = var_cvar_from_returns(sim_normal["port_returns"], alpha=alpha, losses=True)
    risk_boot = var_cvar_from_returns(sim_boot["port_returns"], alpha=alpha, losses=True)
    risk_t = var_cvar_from_returns(sim_t["port_returns"], alpha=alpha, losses=True)

    rows = [
        {
            "method": "normal_mc",
            "alpha": alpha,
            "VaR": risk_normal["VaR"],
            "CVaR": risk_normal["CVaR"],
            "n_sims": n_sims,
            "seed": seed,
            "n_hist": int(returns_df.shape[0]),
            "nu": np.nan,
        },
        {
            "method": "bootstrap_mc",
            "alpha": alpha,
            "VaR": risk_boot["VaR"],
            "CVaR": risk_boot["CVaR"],
            "n_sims": n_sims,
            "seed": seed,
            "n_hist": int(returns_df.shape[0]),
            "nu": np.nan,
        },
        {
            "method": "t_mc",
            "alpha": alpha,
            "VaR": risk_t["VaR"],
            "CVaR": risk_t["CVaR"],
            "n_sims": n_sims,
            "seed": seed,
            "n_hist": int(returns_df.shape[0]),
            "nu": float(nu),
        },
    ]

    res = pd.DataFrame(rows)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)

    return res


def exp03_asymptotics(
    returns_path: str,
    weights: Union[dict, list],
    alpha: float = 0.99,
    conf: float = 0.95,
    seed: int = 0,
    n_grid: list[int] = [250, 500, 1000, 2000],
    reps: int = 200,
    out_summary: str = "paper/tables/exp03_asymptotics_summary.csv",
) -> pd.DataFrame:
    """
    Experiment 03:
    Empirically verify asymptotic normality / n^{-1/2} scaling of VaR estimator.

    Procedure:
      - Treat the full historical sample as "population"
      - Define a high-precision reference VaR using bootstrap resampling from full data
      - For each n in n_grid: sample n rows (with replacement) reps times, compute VaR and CI
      - Summarize RMSE and CI coverage
    """
    returns_df = _load_returns_csv(returns_path)
    rng = np.random.default_rng(seed)

    X_full = returns_df.to_numpy(dtype=float)
    cols = list(returns_df.columns)

    # Convert weights to aligned array by reusing simulate helper indirectly:
    # compute portfolio returns directly from sampled rows.
    if isinstance(weights, dict):
        w = np.array([weights[c] for c in cols], dtype=float)
        w = w / w.sum()
    else:
        w = np.asarray(weights, dtype=float).ravel()
        w = w / w.sum()

    # Full-sample portfolio returns (historical)
    port_full = X_full @ w

    # Reference VaR using large bootstrap from historical portfolio returns
    ref_sims = 200000
    idx_ref = rng.integers(0, port_full.size, size=ref_sims)
    ref_port = port_full[idx_ref]
    ref_var = float(np.quantile(-ref_port, alpha, method="linear"))  # loss quantile

    rows = []
   
    for n in n_grid:
        errs = []
        cover = 0
        avg_width = 0.0
        fhats = []

        for r in range(reps):
            idx = rng.integers(0, port_full.size, size=n)
            sample = port_full[idx]

            qhat = float(np.quantile(-sample, alpha, method="linear"))
            errs.append(qhat - ref_var)

            ci = quantile_asymptotic_ci(sample, alpha=alpha, conf=conf, losses=True)
            fhats.append(ci["fhat"])

            if ci["ci_low"] <= ref_var <= ci["ci_high"]:
                cover += 1
            avg_width += (ci["ci_high"] - ci["ci_low"])
        
        fhats = np.asarray(fhats, dtype=float)
        errs = np.asarray(errs, dtype=float)
        
        rmse = float(np.sqrt(np.mean(errs**2)))
        bias = float(np.mean(errs))
        coverage = float(cover / reps)
        avg_width = float(avg_width / reps)

        fhat_mean = float(np.mean(fhats))
        fhat_min = float(np.min(fhats))
        fhat_max = float(np.max(fhats))
        print(
            f"[exp03] n={n}  fhat(mean/min/max)={fhat_mean:.4e}/{fhat_min:.4e}/{fhat_max:.4e} "
            f"coverage={coverage:.3f} avg_ci_width={avg_width:.6f}"
        )

        rows.append(
            {
                "alpha": alpha,
                "conf": conf,
                "n": int(n),
                "reps": int(reps),
                "ref_var": float(ref_var),
                "rmse": rmse,
                "bias": bias,
                "coverage": coverage,
                "avg_ci_width": avg_width,
                "fhat_mean": fhat_mean,
                "fhat_min": fhat_min,
                "fhat_max": fhat_max,
            }
        )

    res = pd.DataFrame(rows)
    out_path = Path(out_summary)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    return res

def exp04_dkw_bounds(
    returns_path: str,
    weights: Union[dict, list],
    alpha: float = 0.99,
    conf: float = 0.95,
    delta: float = 0.05,
    seed: int = 0,
    n_grid: list[int] = [250, 500, 1000, 2000],
    reps: int = 200,
    out_csv: str = "paper/tables/exp04_dkw_bounds.csv",
) -> pd.DataFrame:
    """
    Experiment 04:
    Compare asymptotic quantile CI (exp03) vs DKW finite-sample bracket.

    For each n, repeat reps times:
      - compute asymptotic CI for VaR
      - compute DKW bracket for VaR
      - check coverage of both against ref_var
      - summarize avg widths + coverages
    """
    returns_df = _load_returns_csv(returns_path)
    rng = np.random.default_rng(seed)

    X_full = returns_df.to_numpy(dtype=float)
    cols = list(returns_df.columns)

    # weights array aligned to columns
    if isinstance(weights, dict):
        w = np.array([weights[c] for c in cols], dtype=float)
        w = w / w.sum()
    else:
        w = np.asarray(weights, dtype=float).ravel()
        w = w / w.sum()

    port_full = X_full @ w

    # Reference VaR via large bootstrap from historical portfolio returns
    ref_sims = 200000
    idx_ref = rng.integers(0, port_full.size, size=ref_sims)
    ref_port = port_full[idx_ref]
    ref_var = float(np.quantile(-ref_port, alpha, method="linear"))

    rows = []
    for n in n_grid:
        cover_asym = 0
        cover_dkw = 0
        avg_width_asym = 0.0
        avg_width_dkw = 0.0

        for r in range(reps):
            idx = rng.integers(0, port_full.size, size=n)
            sample = port_full[idx]

            ci = quantile_asymptotic_ci(sample, alpha=alpha, conf=conf, losses=True)
            if ci["ci_low"] <= ref_var <= ci["ci_high"]:
                cover_asym += 1
            avg_width_asym += (ci["ci_high"] - ci["ci_low"])

            br = dkw_quantile_bracket(sample, alpha=alpha, delta=delta, losses=True)
            if br["q_lo"] <= ref_var <= br["q_hi"]:
                cover_dkw += 1
            avg_width_dkw += br["width"]

        rows.append(
            {
                "alpha": alpha,
                "conf": conf,
                "delta": delta,
                "n": int(n),
                "reps": int(reps),
                "ref_var": float(ref_var),
                "coverage_asym": float(cover_asym / reps),
                "coverage_dkw": float(cover_dkw / reps),
                "avg_width_asym": float(avg_width_asym / reps),
                "avg_width_dkw": float(avg_width_dkw / reps),
            }
        )

    res = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    return res

def exp05_variance_reduction(
    returns_path: str,
    weights: Union[dict, list],
    alpha: float = 0.99,
    seed: int = 0,
    n_sims: int = 20000,
    reps: int = 200,
    out_csv: str = "paper/tables/exp05_variance_reduction.csv",
) -> pd.DataFrame:
    """
    Experiment 05:
    Compare variability of VaR/CVaR estimates under:
      - Plain normal MC
      - Antithetic normal MC

    We run 'reps' independent repetitions and compute sample std / RMSE
    relative to a high-precision reference.
    """
    returns_df = _load_returns_csv(returns_path)
    rng = np.random.default_rng(seed)

    # Reference using very large plain MC (fixed seed for reproducibility)
    ref = simulate_normal_plain(returns_df, weights, n_sims=300000, seed=seed + 999)
    ref_losses = -ref
    ref_var = var_estimator(ref_losses, alpha)
    ref_cvar = cvar_estimator(ref_losses, alpha)

    methods = ["plain", "antithetic"]
    estimates = {m: {"VaR": [], "CVaR": []} for m in methods}

    for r in range(reps):
        s = int(rng.integers(0, 2_000_000_000))

        pr_plain = simulate_normal_plain(returns_df, weights, n_sims=n_sims, seed=s)
        pr_anti = simulate_normal_antithetic(returns_df, weights, n_sims=n_sims, seed=s)

        for name, pr in [("plain", pr_plain), ("antithetic", pr_anti)]:
            L = -pr
            estimates[name]["VaR"].append(var_estimator(L, alpha))
            estimates[name]["CVaR"].append(cvar_estimator(L, alpha))

    rows = []
    for name in methods:
        v = np.asarray(estimates[name]["VaR"], dtype=float)
        c = np.asarray(estimates[name]["CVaR"], dtype=float)

        rows.append(
            {
                "method": name,
                "alpha": alpha,
                "n_sims": n_sims,
                "reps": reps,
                "ref_var": ref_var,
                "ref_cvar": ref_cvar,
                "mean_var": float(v.mean()),
                "std_var": float(v.std(ddof=1)),
                "rmse_var": float(np.sqrt(np.mean((v - ref_var) ** 2))),
                "mean_cvar": float(c.mean()),
                "std_cvar": float(c.std(ddof=1)),
                "rmse_cvar": float(np.sqrt(np.mean((c - ref_cvar) ** 2))),
            }
        )

    res = pd.DataFrame(rows)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_path, index=False)
    return res


def exp06_importance_sampling(
    returns_path: str,
    weights: Union[dict, list],
    alpha: float = 0.99,
    seed: int = 0,
    n_sims: int = 20000,
    reps: int = 200,
    shift_scale: float = 2.5,
    out_csv: str = "paper/tables/exp06_importance_sampling.csv",
) -> pd.DataFrame:
    """
    Compare plain normal MC vs importance sampling (mean shift) for estimating VaR/CVaR.
    Report std and RMSE relative to a large reference.
    """
    returns_df = _load_returns_csv(returns_path)
    rng = np.random.default_rng(seed)

    # Reference: large plain normal MC
    sim_ref = mc_multivariate_normal(returns_df, weights, n_sims=300000, seed=seed + 999)
    ref_losses = -sim_ref["port_returns"]
    ref_var = float(np.quantile(ref_losses, alpha, method="linear"))
    ref_cvar = float(np.mean(ref_losses[ref_losses >= ref_var]))

    plain_var, plain_cvar = [], []
    is_var, is_cvar = [], []

    for _ in range(reps):
        s = int(rng.integers(0, 2_000_000_000))

        sim_plain = mc_multivariate_normal(returns_df, weights, n_sims=n_sims, seed=s)
        Lp = -sim_plain["port_returns"]
        plain_var.append(float(np.quantile(Lp, alpha, method="linear")))
        v = plain_var[-1]
        plain_cvar.append(float(np.mean(Lp[Lp >= v])))

        sim_is = mc_is_normal_shift(
            returns_df, weights, n_sims=n_sims, seed=s, shift_scale=shift_scale
        )
        est = is_var_cvar(sim_is["losses"], sim_is["is_weights"], alpha=alpha)
        is_var.append(est["VaR"])
        is_cvar.append(est["CVaR"])

    plain_var = np.asarray(plain_var)
    plain_cvar = np.asarray(plain_cvar)
    is_var = np.asarray(is_var)
    is_cvar = np.asarray(is_cvar)

    rows = [
        {
            "method": "plain_normal_mc",
            "alpha": alpha,
            "n_sims": n_sims,
            "reps": reps,
            "shift_scale": np.nan,
            "ref_var": ref_var,
            "ref_cvar": ref_cvar,
            "mean_var": float(plain_var.mean()),
            "std_var": float(plain_var.std(ddof=1)),
            "rmse_var": float(np.sqrt(np.mean((plain_var - ref_var) ** 2))),
            "mean_cvar": float(plain_cvar.mean()),
            "std_cvar": float(plain_cvar.std(ddof=1)),
            "rmse_cvar": float(np.sqrt(np.mean((plain_cvar - ref_cvar) ** 2))),
        },
        {
            "method": "importance_sampling",
            "alpha": alpha,
            "n_sims": n_sims,
            "reps": reps,
            "shift_scale": float(shift_scale),
            "ref_var": ref_var,
            "ref_cvar": ref_cvar,
            "mean_var": float(is_var.mean()),
            "std_var": float(is_var.std(ddof=1)),
            "rmse_var": float(np.sqrt(np.mean((is_var - ref_var) ** 2))),
            "mean_cvar": float(is_cvar.mean()),
            "std_cvar": float(is_cvar.std(ddof=1)),
            "rmse_cvar": float(np.sqrt(np.mean((is_cvar - ref_cvar) ** 2))),
        },
    ]
    res = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    return res


def exp07_misspecification(
    returns_path: str,
    weights: Union[dict, list],
    alpha: float = 0.99,
    seed: int = 0,
    T: int = 2500,
    window: int = 500,
    nu_true: float = 5.0,
    nu_model: float = 5.0,
    out_csv: str = "paper/tables/exp07_misspecification.csv",
) -> pd.DataFrame:
    """
    Generate synthetic 'true' returns from multivariate Student-t (nu_true),
    then compute rolling VaR forecasts using:
      A) normal model (misspecified)
      B) t model with nu_model (less misspecified)

    Backtest exception rate + Kupiec LR statistic.
    """
    returns_df = _load_returns_csv(returns_path)
    rng = np.random.default_rng(seed)

    cols = list(returns_df.columns)
    if isinstance(weights, dict):
        w = np.array([weights[c] for c in cols], dtype=float)
        w = w / w.sum()
    else:
        w = np.asarray(weights, dtype=float).ravel()
        w = w / w.sum()

    mu = returns_df.mean(axis=0).to_numpy(dtype=float)
    cov_hist = returns_df.cov(ddof=1).to_numpy(dtype=float)

    # Convert historical covariance to Student-t scale matrix Σ for the DGP:
    Sigma_true = cov_hist * ((nu_true - 2.0) / nu_true)

    # True synthetic path
    X_true = simulate_multivariate_t(mu=mu, cov=Sigma_true, nu=nu_true, n_sims=T, rng=rng)
    port_true = X_true @ w
    losses_true = -port_true

    # Rolling forecasts
    var_norm = []
    var_t = []
    realized = []

    for t in range(window, T):
        hist = X_true[t - window : t, :]
        # fit normal
        mu_h = hist.mean(axis=0)
        cov_h = np.cov(hist, rowvar=False, ddof=1)

        # forecast distribution of portfolio return
        mean_p = float(mu_h @ w)
        var_p = float(w @ cov_h @ w)
        std_p = float(np.sqrt(max(var_p, 1e-18)))

        # Normal VaR for loss: L = -R
        # VaR_alpha(L) = -(mean_p + z_{1-alpha} * std_p) ??? 
        # Want alpha-quantile of loss => alpha-quantile of (-R) = - (1-alpha)-quantile of R.
        # So use q_R = mean + z_{1-alpha}*std; VaR_L = -q_R
        z = inv_norm_cdf(1.0 - alpha)  # negative number
        qR = mean_p + z * std_p
        var_norm.append(float(-qR))

        # Student-t approximation (scale correction):
        # If X ~ t_nu(mu, Sigma), then Cov(X) = nu/(nu-2) * Sigma.
        # Here cov_h estimates Cov(X), so Sigma_hat = (nu-2)/nu * cov_h.
        Sigma_hat = cov_h * ((nu_model - 2.0) / nu_model)

        var_p_t = float(w @ Sigma_hat @ w)
        std_p_t = float(np.sqrt(max(var_p_t, 1e-18)))

        tcrit = t_quantile(1.0 - alpha, nu_model)
        qR_t = mean_p + tcrit * std_p_t
        var_t.append(float(-qR_t))

        realized.append(float(losses_true[t]))

    realized = np.asarray(realized)
    var_norm = np.asarray(var_norm)
    var_t = np.asarray(var_t)

    exc_norm = (realized > var_norm).astype(int)
    exc_t = (realized > var_t).astype(int)

    hit_norm = float(exc_norm.mean())
    hit_t = float(exc_t.mean())

    lr_norm, p_norm = kupiec_uc_from_exceptions(exc_norm, alpha=alpha)
    lr_t, p_t = kupiec_uc_from_exceptions(exc_t, alpha=alpha)

    if nu_model >= 200:
        t_label = f"t_model_df{float(nu_model):g}_approx_normal"
    else:
        t_label = f"t_model_df{float(nu_model):g}"
    
    res = pd.DataFrame(
        [
            {
                "model": "normal_misspecified",
                "alpha": alpha,
                "T": T,
                "window": window,
                "nu_true": nu_true,
                "nu_model": np.nan,
                "exception_rate": hit_norm,
                "expected_rate": 1 - alpha,
                "kupiec_lr": lr_norm,
                "kupiec_crit_5pct": 3.841,  # ChiSq(1) 95% quantile
                "reject_5pct": bool(lr_norm > 3.841),
            },
            {
                "model": t_label,
                "alpha": alpha,
                "T": T,
                "window": window,
                "nu_true": nu_true,
                "nu_model": nu_model,
                "exception_rate": hit_t,
                "expected_rate": 1 - alpha,
                "kupiec_lr": lr_t,
                "kupiec_crit_5pct": 3.841,
                "reject_5pct": bool(lr_t > 3.841),
            },
        ]
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    return res


def exp08_es_backtesting(
    returns_path: str,
    weights: Union[dict, list],
    alpha: float = 0.99,
    window: int = 500,
    seed: int = 0,
    out_csv: str = "paper/tables/exp08_es_backtesting.csv",
) -> pd.DataFrame:
    """
    Rolling VaR/ES forecasts under:
      - Normal (parametric)
      - Historical (nonparametric)

    Adds Kupiec UC + Christoffersen independence tests.
    """
    returns_df = _load_returns_csv(returns_path)
    cols = list(returns_df.columns)

    if isinstance(weights, dict):
        w = np.array([weights[c] for c in cols], dtype=float)
        w = w / w.sum()
    else:
        w = np.asarray(weights, dtype=float).ravel()
        w = w / w.sum()

    X = returns_df.dropna().to_numpy(dtype=float)
    port_r = X @ w
    losses = -port_r

    V_norm, ES_norm = [], []
    V_hist, ES_hist = [], []
    realized = []

    for t in range(window, len(losses)):
        hist_L = losses[t - window : t]

        # Historical VaR/ES (loss space)
        v_h = float(np.quantile(hist_L, alpha, method="linear"))
        es_h = float(hist_L[hist_L >= v_h].mean())
        V_hist.append(v_h)
        ES_hist.append(es_h)

        # Normal VaR/ES (fit on returns, then transform)
        r_hist = -hist_L
        mu = float(np.mean(r_hist))
        sig = float(np.std(r_hist, ddof=1))
        sig = max(sig, 1e-12)

        z = inv_norm_cdf(1.0 - alpha)  # negative
        qR = mu + z * sig
        v_n = float(-qR)

        phi = float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi))
        es_n = float(-(mu - sig * (phi / (1.0 - alpha))))

        V_norm.append(v_n)
        ES_norm.append(es_n)

        realized.append(float(losses[t]))

    realized = np.asarray(realized, dtype=float)
    V_norm = np.asarray(V_norm, dtype=float)
    ES_norm = np.asarray(ES_norm, dtype=float)
    V_hist = np.asarray(V_hist, dtype=float)
    ES_hist = np.asarray(ES_hist, dtype=float)

    s_norm = es_backtest_summary(realized, V_norm, ES_norm, alpha)
    s_hist = es_backtest_summary(realized, V_hist, ES_hist, alpha)

    # Add UC + IND tests
    exc_norm = s_norm.pop("exceptions")
    exc_hist = s_hist.pop("exceptions")

    # UC
    lr, p = kupiec_uc_from_exceptions(exc_norm, alpha=alpha)
    s_norm["kupiec_lr_uc"] = lr
    s_norm["kupiec_p_uc"] = p

    lr, p = kupiec_uc_from_exceptions(exc_hist, alpha=alpha)
    s_hist["kupiec_lr_uc"] = lr
    s_hist["kupiec_p_uc"] = p

    # IND
    lr, p = christoffersen_ind_test(exc_norm)
    s_norm["christoffersen_lr_ind"] = lr
    s_norm["christoffersen_p_ind"] = p

    lr, p = christoffersen_ind_test(exc_hist)
    s_hist["christoffersen_lr_ind"] = lr
    s_hist["christoffersen_p_ind"] = p

    # 5% critical values (ChiSq(1))
    crit_5 = 3.841
    s_norm["crit_5pct"] = crit_5
    s_hist["crit_5pct"] = crit_5
    s_norm["reject_uc_5pct"] = bool(s_norm["kupiec_lr_uc"] > crit_5)
    s_hist["reject_uc_5pct"] = bool(s_hist["kupiec_lr_uc"] > crit_5)
    s_norm["reject_ind_5pct"] = bool(s_norm["christoffersen_lr_ind"] > crit_5)
    s_hist["reject_ind_5pct"] = bool(s_hist["christoffersen_lr_ind"] > crit_5)
    s_norm["fz0_mean"] = mean_fz0(realized, V_norm, ES_norm, alpha)
    s_hist["fz0_mean"] = mean_fz0(realized, V_hist, ES_hist, alpha)  

    res = pd.DataFrame(
        [
            {"model": "normal", "alpha": alpha, "window": window, **s_norm},
            {"model": "historical", "alpha": alpha, "window": window, **s_hist},
        ]
    )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    return res


def exp09_garch_var_es(
    returns_path: str,
    weights: Union[dict, list],
    alpha: float = 0.99,
    window: int = 500,
    seed: int = 0,
    out_csv: str = "paper/tables/exp09_garch_var_es.csv",
) -> pd.DataFrame:
    """
    Rolling VaR/ES using GARCH(1,1) volatility forecasts (conditional normal).
    Adds Kupiec UC + Christoffersen independence tests.
    """
    returns_df = _load_returns_csv(returns_path)
    cols = list(returns_df.columns)

    if isinstance(weights, dict):
        w = np.array([weights[c] for c in cols], dtype=float)
        w = w / w.sum()
    else:
        w = np.asarray(weights, dtype=float).ravel()
        w = w / w.sum()

    X = returns_df.dropna().to_numpy(dtype=float)
    port_r = X @ w
    losses = -port_r

    V, ES, realized = [], [], []

    for t in range(window, len(port_r)):
        r_hist = port_r[t - window : t]

        res = fit_garch11_arch(r_hist, dist="normal")
        mu = garch_mean(res)
        sig = garch_forecast_sigma(res)

        z = inv_norm_cdf(1.0 - alpha)
        qR = mu + z * sig
        v = float(-qR)

        phi = float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi))
        es = float(-(mu - sig * (phi / (1.0 - alpha))))

        V.append(v)
        ES.append(es)
        realized.append(float(losses[t]))

    realized = np.asarray(realized)
    V = np.asarray(V)
    ES = np.asarray(ES)

    s = es_backtest_summary(realized, V, ES, alpha)
    exc = s.pop("exceptions")

    lr, p = kupiec_uc_from_exceptions(exc, alpha=alpha)
    s["kupiec_lr_uc"] = lr
    s["kupiec_p_uc"] = p

    lr, p = christoffersen_ind_test(exc)
    s["christoffersen_lr_ind"] = lr
    s["christoffersen_p_ind"] = p

    crit_5 = 3.841
    s["crit_5pct"] = crit_5
    s["reject_uc_5pct"] = bool(s["kupiec_lr_uc"] > crit_5)
    s["reject_ind_5pct"] = bool(s["christoffersen_lr_ind"] > crit_5)
    s["fz0_mean"] = mean_fz0(realized, V, ES, alpha)

    resdf = pd.DataFrame([{"model": "garch11_normal", "alpha": alpha, "window": window, **s}])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    resdf.to_csv(out_csv, index=False)
    return resdf

def exp10_garch_t_var_es(
    returns_path: str,
    weights: Union[dict, list],
    alpha: float = 0.99,
    window: int = 500,
    seed: int = 0,
    out_csv: str = "paper/tables/exp10_garch_t_var_es.csv",
) -> pd.DataFrame:
    """
    Rolling VaR/ES using GARCH(1,1) with Student-t innovations (arch dist='t').

    Outputs:
      - summary CSV (out_csv)
      - series CSV alongside it: paper/tables/exp10_garch_t_series.csv
        containing time series of sigma, nu, VaR, ES, realized loss, exceptions
    """

    returns_df = _load_returns_csv(returns_path)
    cols = list(returns_df.columns)

    if isinstance(weights, dict):
        w = np.array([weights[c] for c in cols], dtype=float)
        w = w / w.sum()
    else:
        w = np.asarray(weights, dtype=float).ravel()
        w = w / w.sum()

    X = returns_df.dropna().to_numpy(dtype=float)
    port_r = X @ w
    losses = -port_r

    V, ES, realized = [], [], []
    sig_series, nu_series = [], []
    idx_series = returns_df.dropna().index[window:]  # aligns with realized after rolling

    import math

    for t in range(window, len(port_r)):
        r_hist = port_r[t - window : t]

        res = fit_garch11_arch(r_hist, dist="t")
        mu = garch_mean(res)
        sig = garch_forecast_sigma(res)
        nu = garch_df_if_t(res)
        if nu is None:
            nu = 10.0

        # Student-t quantile for returns (left tail p=1-alpha)
        tcrit = t_quantile(1.0 - alpha, float(nu))  # negative
        qR = mu + tcrit * sig
        v = float(-qR)

        # Student-t ES for returns left tail (p = 1-alpha)
        p = 1.0 - alpha
        x = float(tcrit)

        # pdf standard t
        num = math.gamma((nu + 1.0) / 2.0)
        den = math.sqrt(nu * math.pi) * math.gamma(nu / 2.0)
        ft = (num / den) * (1.0 + (x * x) / nu) ** (-(nu + 1.0) / 2.0)

        # E[R | R <= q_p] = mu - sig * ((nu + x^2)/((nu-1)*p)) * f_t(x)  (nu>1)
        if nu <= 1:
            es_r_left = float("nan")
        else:
            es_r_left = mu - sig * ((nu + x * x) / ((nu - 1.0) * p)) * ft

        es = float(-es_r_left)  # ES for losses

        V.append(v)
        ES.append(es)
        realized.append(float(losses[t]))
        sig_series.append(float(sig))
        nu_series.append(float(nu))

    realized = np.asarray(realized, dtype=float)
    V = np.asarray(V, dtype=float)
    ES = np.asarray(ES, dtype=float)
    sig_series = np.asarray(sig_series, dtype=float)
    nu_series = np.asarray(nu_series, dtype=float)

    # Summary diagnostics
    s = es_backtest_summary(realized, V, ES, alpha)
    exc = s.pop("exceptions")

    lr_uc, p_uc = kupiec_uc_from_exceptions(exc, alpha=alpha)
    lr_ind, p_ind = christoffersen_ind_test(exc)

    crit_5 = 3.841
    s["kupiec_lr_uc"] = lr_uc
    s["kupiec_p_uc"] = p_uc
    s["christoffersen_lr_ind"] = lr_ind
    s["christoffersen_p_ind"] = p_ind
    s["crit_5pct"] = crit_5
    s["reject_uc_5pct"] = bool(lr_uc > crit_5)
    s["reject_ind_5pct"] = bool(lr_ind > crit_5)

    # nu/sigma stats
    s["nu_mean"] = float(np.mean(nu_series))
    s["nu_min"] = float(np.min(nu_series))
    s["nu_max"] = float(np.max(nu_series))
    s["sigma_mean"] = float(np.mean(sig_series))
    s["sigma_min"] = float(np.min(sig_series))
    s["sigma_max"] = float(np.max(sig_series))
    s["fz0_mean"] = mean_fz0(realized, V, ES, alpha)

    resdf = pd.DataFrame(
        [
            {
                "model": "garch11_t",
                "alpha": alpha,
                "window": window,
                "nu_hat_meaning": "arch nu",
                **s,
            }
        ]
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    resdf.to_csv(out_csv, index=False)

    # Save series table
    series_path = Path(out_csv).with_name("exp10_garch_t_series.csv")
    series_df = pd.DataFrame(
        {
            "sigma": sig_series,
            "nu": nu_series,
            "VaR": V,
            "ES": ES,
            "RealizedLoss": realized,
            "Exception": exc.astype(int),
        },
        index=idx_series,
    )
    series_df.index.name = "Date"
    series_df.to_csv(series_path, index=True)

    return resdf

def main():
    parser = argparse.ArgumentParser(description="Run Monte Carlo VaR research experiments.")

    parser.add_argument(
        "--exp",
        required=True,
        choices=["exp01", "exp02", "exp03", "exp04", "exp05", "exp06", "exp07", "exp08", "exp09", "exp10"],
        help="Experiment name.",
    )

    parser.add_argument("--returns", required=True, help="Path to returns CSV.")

    parser.add_argument(
        "--weights",
        default=None,
        help='Portfolio weights as JSON dict or list. Example: \'{"AAPL":0.5,"MSFT":0.5}\' or "[0.5,0.5]"',
    )
    parser.add_argument(
        "--weights-file",
        default=None,
        help="Path to JSON file containing weights dict or list (recommended on Windows).",
    )

    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--n_sims", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)

    # exp02: Student-t df
    parser.add_argument(
        "--nu",
        type=float,
        default=None,
        help="Student-t df for exp02. If omitted, auto-select.",
    )

    # exp03/exp04: asymptotics + DKW settings
    parser.add_argument(
        "--conf",
        type=float,
        default=0.95,
        help="Confidence level for asymptotic CI (exp03/exp04).",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="DKW failure probability delta (exp04).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=200,
        help="Number of repetitions (exp03/exp04/exp05/exp06).",
    )
    parser.add_argument(
        "--n_grid",
        type=str,
        default="250,500,1000,2000",
        help='Comma-separated sample sizes (exp03/exp04), e.g. "250,500,1000,2000".',
    )

    # exp06: importance sampling
    parser.add_argument(
        "--shift_scale",
        type=float,
        default=2.5,
        help="Mean-shift scale for importance sampling (exp06).",
    )

    # exp07: misspecification study settings
    parser.add_argument("--T", type=int, default=2500, help="Length of synthetic series (exp07).")
    parser.add_argument("--window", type=int, default=500, help="Rolling window size (exp07/exp08/exp09).")
    parser.add_argument("--nu_true", type=float, default=5.0, help="True df for Student-t DGP (exp07).")
    parser.add_argument("--nu_model", type=float, default=5.0, help="Model df for Student-t forecast (exp07).")

    parser.add_argument("--out", default=None, help="Output CSV path. Defaults depend on experiment.")

    args = parser.parse_args()
    weights = _parse_weights(args.weights, args.weights_file)

    if args.exp == "exp01":
        out = args.out or "paper/tables/exp01_classical_var.csv"
        df = exp01_classical_var(
            returns_path=args.returns,
            weights=weights,
            n_sims=args.n_sims,
            alpha=args.alpha,
            seed=args.seed,
            out_csv=out,
        )
        print(df.to_string(index=False))

    elif args.exp == "exp02":
        out = args.out or "paper/tables/exp02_normal_vs_t.csv"
        df = exp02_normal_vs_t(
            returns_path=args.returns,
            weights=weights,
            n_sims=args.n_sims,
            alpha=args.alpha,
            seed=args.seed,
            nu=args.nu,
            out_csv=out,
        )
        print(df.to_string(index=False))

    elif args.exp == "exp03":
        out = args.out or "paper/tables/exp03_asymptotics_summary.csv"
        try:
            n_grid = [int(x.strip()) for x in args.n_grid.split(",") if x.strip()]
        except ValueError:
            raise ValueError('n_grid must be comma-separated integers, e.g. "250,500,1000".')

        df = exp03_asymptotics(
            returns_path=args.returns,
            weights=weights,
            alpha=args.alpha,
            conf=args.conf,
            seed=args.seed,
            n_grid=n_grid,
            reps=args.reps,
            out_summary=out,
        )
        print(df.to_string(index=False))

    elif args.exp == "exp04":
        out = args.out or "paper/tables/exp04_dkw_bounds.csv"
        try:
            n_grid = [int(x.strip()) for x in args.n_grid.split(",") if x.strip()]
        except ValueError:
            raise ValueError('n_grid must be comma-separated integers, e.g. "250,500,1000".')

        df = exp04_dkw_bounds(
            returns_path=args.returns,
            weights=weights,
            alpha=args.alpha,
            conf=args.conf,
            delta=args.delta,
            seed=args.seed,
            n_grid=n_grid,
            reps=args.reps,
            out_csv=out,
        )
        print(df.to_string(index=False))

    elif args.exp == "exp05":
        out = args.out or "paper/tables/exp05_variance_reduction.csv"
        df = exp05_variance_reduction(
            returns_path=args.returns,
            weights=weights,
            alpha=args.alpha,
            seed=args.seed,
            n_sims=args.n_sims,
            reps=args.reps,
            out_csv=out,
        )
        print(df.to_string(index=False))

    elif args.exp == "exp06":
        out = args.out or "paper/tables/exp06_importance_sampling.csv"
        df = exp06_importance_sampling(
            returns_path=args.returns,
            weights=weights,
            alpha=args.alpha,
            seed=args.seed,
            n_sims=args.n_sims,
            reps=args.reps,
            shift_scale=args.shift_scale,
            out_csv=out,
        )
        print(df.to_string(index=False))

    elif args.exp == "exp07":
        out = args.out or "paper/tables/exp07_misspecification.csv"
        df = exp07_misspecification(
            returns_path=args.returns,
            weights=weights,
            alpha=args.alpha,
            seed=args.seed,
            T=args.T,
            window=args.window,
            nu_true=args.nu_true,
            nu_model=args.nu_model,
            out_csv=out,
        )
        print(df.to_string(index=False))

    elif args.exp == "exp08":
        out = args.out or "paper/tables/exp08_es_backtesting.csv"
        df = exp08_es_backtesting(
            returns_path=args.returns,
            weights=weights,
            alpha=args.alpha,
            window=args.window,
            seed=args.seed,
            out_csv=out,
        )
        print(df.to_string(index=False))

    elif args.exp == "exp09":
        out = args.out or "paper/tables/exp09_garch_var_es.csv"
        df = exp09_garch_var_es(
            returns_path=args.returns,
            weights=weights,
            alpha=args.alpha,
            window=args.window,
            seed=args.seed,
            out_csv=out,
        )
        print(df.to_string(index=False))

    elif args.exp == "exp10":
        out = args.out or "paper/tables/exp10_garch_t_var_es.csv"
        df = exp10_garch_t_var_es(
            returns_path=args.returns,
            weights=weights,
            alpha=args.alpha,
            window=args.window,
            seed=args.seed,
            out_csv=out,
        )
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()