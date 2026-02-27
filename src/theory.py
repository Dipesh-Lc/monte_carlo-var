from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np
from math import sqrt, pi, exp
from scipy.stats import t

def _silverman_bandwidth(x: np.ndarray) -> float:
    """
    Silverman's rule-of-thumb bandwidth for KDE.
    Works fine for 1D KDE in this project.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return 1.0
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    if sigma <= 1e-12:
        sigma = std if std > 1e-12 else 1.0
    h = 0.9 * sigma * (n ** (-1 / 5))
    return float(max(h, 1e-12))


def kde_density_at_point(x: np.ndarray, point: float, bandwidth: float | None = None) -> float:
    """
    Gaussian KDE density estimate at a single point.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 10:
        raise ValueError("Need at least 10 samples for KDE density estimation.")

    h = _silverman_bandwidth(x) if bandwidth is None else float(bandwidth)
    u = (point - x) / h
    # Gaussian kernel
    kern = np.exp(-0.5 * u * u) / sqrt(2 * pi)
    fhat = float(np.mean(kern) / h)
    return max(fhat, 1e-300)


def quantile_asymptotic_ci(
    samples: np.ndarray,
    alpha: float,
    conf: float = 0.95,
    losses: bool = True,
    density_method: str = "kde",
) -> Dict[str, Any]:
    """
    Asymptotic CI for the alpha-quantile of losses based on:
      sqrt(n)(qhat - q) -> N(0, alpha(1-alpha)/f(q)^2)

    Returns dict with:
      qhat, se, ci_low, ci_high, fhat, n, alpha, conf
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")
    if not (0.0 < conf < 1.0):
        raise ValueError("conf must be in (0,1).")

    x = np.asarray(samples, dtype=float).ravel()
    if x.size < 50:
        raise ValueError("Need at least 50 samples for asymptotic CI to be meaningful.")
    L = -x if losses else x

    qhat = float(np.quantile(L, alpha, method="linear"))
    n = L.size

    if density_method != "kde":
        raise ValueError("Only density_method='kde' is implemented in this project.")

    fhat = kde_density_at_point(L, qhat)

    se = sqrt(alpha * (1 - alpha) / (n * (fhat ** 2)))

    # Normal critical value : I am using approximation for z
    # For 95% conf, z≈1.959963984540054; for general conf, approximate via inverse error function.
    # I'll implement a small approximation for inverse normal CDF.
    z = z_from_conf(conf)

    ci_low = qhat - z * se
    ci_high = qhat + z * se

    return {
        "qhat": qhat,
        "se": float(se),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "fhat": float(fhat),
        "n": int(n),
        "alpha": float(alpha),
        "conf": float(conf),
        "z": float(z),
    }


def z_from_conf(conf: float) -> float:
    """
    Approximate z_{1-(1-conf)/2} without scipy.
    Uses an approximation to inverse normal CDF.
    Good enough for research plots/tables.
    """
    p = 1 - (1 - conf) / 2  # e.g. 0.975 for 95%
    return float(inv_norm_cdf(p))


def inv_norm_cdf(p: float) -> float:
    """
    Rational approximation for inverse normal CDF (Acklam-style).
    Accurate for (0,1).
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0,1)")

    # Coefficients in rational approximations
    a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ]
    d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = sqrt(-2 * np.log(p))
        num = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        return num / den

    if p > phigh:
        q = sqrt(-2 * np.log(1 - p))
        num = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5])
        den = ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
        return num / den

    q = p - 0.5
    r = q * q
    num = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q
    den = (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    return num / den

def t_quantile(p: float, nu: float) -> float:
    return float(t.ppf(p, df=nu))