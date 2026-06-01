"""Lightweight statistics (no SciPy dependency).

* Mann-Whitney U test with normal-approximation p-value.
* Wilson 95% confidence interval for a binomial proportion.
* Two-proportion z-test.
"""
import math
from typing import List, Tuple


def _phi(x: float) -> float:
    """CDF of standard normal via erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mann_whitney_u(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Return (U, two-sided p-value).

    Normal approximation; for the sample sizes used here (~50)
    the error is negligible.
    """
    n_x, n_y = len(x), len(y)
    combined = [(v, 0) for v in x] + [(v, 1) for v in y]
    combined.sort(key=lambda t: t[0])
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_x = sum(r for r, (_, lbl) in zip(ranks, combined) if lbl == 0)
    u_x = rank_sum_x - n_x * (n_x + 1) / 2
    u_y = n_x * n_y - u_x
    u = min(u_x, u_y)
    mean = n_x * n_y / 2.0
    sigma = math.sqrt(n_x * n_y * (n_x + n_y + 1) / 12.0)
    z = (u - mean) / sigma if sigma > 0 else 0.0
    p = 2 * (1 - _phi(abs(z)))
    return u, p


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    centre = (p_hat + z * z / (2 * n)) / denom
    half = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def two_proportion_z(s_a: int, n_a: int, s_b: int, n_b: int) -> Tuple[float, float]:
    """Two-proportion z-test; returns (z, two-sided p-value)."""
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    p_a = s_a / n_a; p_b = s_b / n_b
    p_pool = (s_a + s_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_a - p_b) / se
    p = 2 * (1 - _phi(abs(z)))
    return z, p
