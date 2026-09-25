"""core/_stats.py — Student t distribution with the standard library only.

The CDF uses the regularized incomplete beta function (continued fraction,
Numerical Recipes §6.4); the quantile inverts it with safeguarded Newton. Accuracy is
~1e-10, far beyond what a decision rule needs, and keeps core/ free of scipy.
"""
from __future__ import annotations

import math
from statistics import NormalDist

_EPS = 1e-14
_MAX_ITER = 300


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, _MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    if df <= 0:
        raise ValueError("df must be positive")
    x = df / (df + t * t)
    tail = 0.5 * _betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_pdf(t: float, df: float) -> float:
    ln = (math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * math.log(df * math.pi)
          - (df + 1) / 2 * math.log1p(t * t / df))
    return math.exp(ln)


def t_ppf(p: float, df: float) -> float:
    """Quantile of Student's t: the t such that P(T <= t) = p.

    Newton steps from the normal quantile, kept inside a shrinking bisection bracket
    so a bad step can never escape (heavy tails at df ≈ 1 make pure Newton unsafe).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        lo, hi = hi, hi * 2.0
    x = min(max(NormalDist().inv_cdf(p), lo), hi)
    for _ in range(100):
        f = t_cdf(x, df) - p
        if abs(f) < 1e-13:
            return x
        if f < 0:
            lo = x
        else:
            hi = x
        step = x - f / t_pdf(x, df)
        x = step if lo < step < hi else (lo + hi) / 2.0
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    return x
