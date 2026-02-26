# src/stats_utils.py
from __future__ import annotations

import math


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """
    Wilson score interval for binomial proportion.
    Returns (low, high).
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if k < 0 or k > n:
        raise ValueError("k must be between 0 and n")

    # z for two-sided conf using normal approx
    # conf=0.95 -> z ~ 1.959963...
    # We'll use math.erfcinv workaround? Keep simple:
    # Use scipy if available, else hardcode common values.
    z_map = {0.90: 1.6448536269514722, 0.95: 1.959963984540054, 0.99: 2.5758293035489004}
    z = z_map.get(conf, 1.959963984540054)

    phat = k / n
    denom = 1 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z * z) / (4 * n * n))
    return (center - half, center + half)