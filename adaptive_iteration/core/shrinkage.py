"""core/shrinkage.py — Correcting reported effects for the winner's curse.

The experiments that get declared winners are disproportionately the ones noise
happened to push upward, so their measured effects overstate the truth on average.
Shrinkage pulls each estimate toward zero by an amount that depends on how noisy it
is: a precise estimate barely moves, a vague one moves a lot.

Model (empirical Bayes, normal–normal): true effects across a domain's experiments
are spread around 0 with standard deviation tau ("most changes do little"), and each
measured effect is the true one plus noise with standard error se. Then the best
estimate of the true effect is

    shrunk = effect × tau² / (tau² + se²)

tau is not guessed: it is estimated from the domain's own closed experiments as
sqrt(max(0, mean(effect²) − mean(se²))) — the spread of measured effects beyond what
their noise explains. With too few closed experiments there is no estimate and no
correction. Verdicts are never changed; this only affects the effect size reported.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from statistics import NormalDist
from typing import Iterable, Optional

from .decision import Decision

MIN_EXPERIMENTS = 5


def approx_se(decision: Decision) -> Optional[float]:
    """Standard error implied by a decision's interval (normal approximation; for
    small-sample t intervals it errs slightly large, i.e. toward more shrinkage)."""
    if decision.interval is None or decision.confidence is None:
        return None
    lo, hi = decision.interval
    z = NormalDist().inv_cdf(1 - (1 - decision.confidence) / 2)
    se = (hi - lo) / (2 * z)
    return se if se > 0 else None


@dataclass(frozen=True)
class Prior:
    tau: Optional[float]          # SD of true effects across the domain; None = unknown
    n_experiments: int
    basis: str

    def shrink(self, effect: Optional[float], se: Optional[float]) -> Optional[float]:
        if effect is None or se is None or self.tau is None:
            return None
        t2 = self.tau ** 2
        return effect * t2 / (t2 + se ** 2) if t2 + se ** 2 > 0 else 0.0


def estimate_prior(decisions: Iterable[Decision]) -> Prior:
    pairs = [(d.effect, approx_se(d)) for d in decisions
             if d.outcome.is_final and d.effect is not None]
    pairs = [(e, s) for e, s in pairs if s is not None and math.isfinite(e)]
    n = len(pairs)
    if n < MIN_EXPERIMENTS:
        return Prior(None, n, f"only {n} closed experiments with an interval "
                              f"(need {MIN_EXPERIMENTS}); no correction")
    excess = statistics.fmean(e * e for e, _ in pairs) - statistics.fmean(s * s for _, s in pairs)
    tau = math.sqrt(max(0.0, excess))
    return Prior(tau, n, f"true-effect spread {tau:.3g} estimated from {n} closed experiments")
