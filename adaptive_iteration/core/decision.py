"""core/decision.py — Decision types and the DecisionRule strategy.

A DecisionRule looks at two validated samples and says which arm is better, that
they are practically equivalent, or that there is not enough evidence yet. The
Evaluator (core/evaluator.py) decides *when* to ask and which data counts; the rule
only decides *what* the data says.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from statistics import NormalDist
from typing import Any, Optional, Protocol

from ._stats import t_ppf
from .metrics import MetricSpec


class Outcome(str, Enum):
    A_BETTER = "a_better"
    B_BETTER = "b_better"
    EQUIVALENT = "equivalent"                  # difference is confidently below min_effect
    INSUFFICIENT = "insufficient"              # not known yet; keep collecting
    NO_DETECTABLE_DIFF = "no_detectable_diff"  # last checkpoint reached, still undecided

    @property
    def is_final(self) -> bool:
        return self is not Outcome.INSUFFICIENT


@dataclass(frozen=True)
class Sample:
    """Valid, mature values for one arm. pair_ids align with values (paired mode)."""
    values: tuple[float, ...]
    unit_ids: tuple[str, ...] = ()
    pair_ids: tuple[Optional[str], ...] = ()

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class DecisionContext:
    experiment_id: str
    paired: bool
    checkpoint: int          # 1-based
    max_checkpoints: int

    @property
    def is_last(self) -> bool:
        return self.checkpoint >= self.max_checkpoints


@dataclass(frozen=True)
class RuleResult:
    """What a DecisionRule returns. effect/interval are oriented so that positive
    means B is better, regardless of MetricSpec.higher_is_better."""
    outcome: Outcome
    reason: str
    effect: Optional[float] = None
    interval: Optional[tuple[float, float]] = None
    confidence: Optional[float] = None
    needed_n: Optional[int] = None
    params: dict[str, Any] = field(default_factory=dict)


class DecisionRule(Protocol):
    name: str

    def decide(
        self, a: Sample, b: Sample, spec: MetricSpec, ctx: DecisionContext
    ) -> RuleResult: ...


@dataclass(frozen=True)
class Decision:
    experiment_id: str
    outcome: Outcome
    metric: str
    checkpoint: int
    n_a: int
    n_b: int
    excluded: dict[str, int]
    rule: str
    rule_params: dict[str, Any]
    reason: str
    decided_at: str
    effect: Optional[float] = None
    interval: Optional[tuple[float, float]] = None
    confidence: Optional[float] = None
    needed_n: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "outcome": self.outcome.value,
            "metric": self.metric,
            "checkpoint": self.checkpoint,
            "n_a": self.n_a,
            "n_b": self.n_b,
            "excluded": dict(self.excluded),
            "rule": self.rule,
            "rule_params": dict(self.rule_params),
            "reason": self.reason,
            "decided_at": self.decided_at,
            "effect": self.effect,
            "interval": list(self.interval) if self.interval else None,
            "confidence": self.confidence,
            "needed_n": self.needed_n,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Decision":
        return cls(
            experiment_id=d["experiment_id"],
            outcome=Outcome(d["outcome"]),
            metric=d["metric"],
            checkpoint=d["checkpoint"],
            n_a=d["n_a"],
            n_b=d["n_b"],
            excluded=dict(d.get("excluded", {})),
            rule=d["rule"],
            rule_params=dict(d.get("rule_params", {})),
            reason=d.get("reason", ""),
            decided_at=d["decided_at"],
            effect=d.get("effect"),
            interval=tuple(d["interval"]) if d.get("interval") else None,
            confidence=d.get("confidence"),
            needed_n=d.get("needed_n"),
        )


@dataclass(frozen=True)
class WelchIntervalRule:
    """Default rule: confidence interval for mean(B) − mean(A) compared with ±min_effect.

    - interleaved: Welch t interval (unequal variances)
    - paired: one-sample t interval on per-pair differences
    - alpha is Bonferroni-split over max_checkpoints, so checking every window
      keeps the experiment-wide false-positive rate at or below alpha.
    """
    alpha: float = 0.05
    min_n: int = 5
    power: float = 0.8
    name: str = "welch_interval"

    def __post_init__(self) -> None:
        if self.min_n < 2:
            raise ValueError("min_n must be at least 2 (a variance needs two values)")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")

    def decide(self, a: Sample, b: Sample, spec: MetricSpec, ctx: DecisionContext) -> RuleResult:
        per_check_alpha = self.alpha / ctx.max_checkpoints
        confidence = 1.0 - per_check_alpha
        params = {"alpha": self.alpha, "min_n": self.min_n, "power": self.power,
                  "per_check_alpha": per_check_alpha, "min_effect": spec.min_effect}
        sign = 1.0 if spec.higher_is_better else -1.0

        undecided = Outcome.NO_DETECTABLE_DIFF if ctx.is_last else Outcome.INSUFFICIENT

        if ctx.paired:
            diffs = _paired_diffs(a, b)
            n = len(diffs)
            if n < self.min_n:
                return RuleResult(undecided, f"{n} complete pairs < min_n={self.min_n}",
                                  confidence=confidence, params=params,
                                  needed_n=self.min_n - n)
            mean = statistics.fmean(diffs) * sign
            sd = statistics.stdev(diffs)
            se = sd / math.sqrt(n)
            df = n - 1
            per_arm_sd = sd  # used for needed_n; paired design needs n pairs
        else:
            n_a, n_b = len(a), len(b)
            if min(n_a, n_b) < self.min_n:
                return RuleResult(undecided,
                                  f"n_a={n_a}, n_b={n_b}; each arm needs ≥ min_n={self.min_n}",
                                  confidence=confidence, params=params,
                                  needed_n=self.min_n - min(n_a, n_b))
            var_a, var_b = statistics.variance(a.values), statistics.variance(b.values)
            mean = (statistics.fmean(b.values) - statistics.fmean(a.values)) * sign
            se2 = var_a / n_a + var_b / n_b
            se = math.sqrt(se2)
            if se2 > 0:
                df = se2 ** 2 / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
            else:
                df = n_a + n_b - 2
            per_arm_sd = math.sqrt((var_a + var_b) / 2.0)

        half = t_ppf(1.0 - per_check_alpha / 2.0, df) * se
        lo, hi = mean - half, mean + half
        rope = spec.min_effect
        common = dict(effect=mean, interval=(lo, hi), confidence=confidence, params=params)

        if lo > rope:
            return RuleResult(Outcome.B_BETTER,
                              f"B better: whole interval [{lo:.3g}, {hi:.3g}] above +{rope:g}", **common)
        if hi < -rope:
            return RuleResult(Outcome.A_BETTER,
                              f"A better: whole interval [{lo:.3g}, {hi:.3g}] below −{rope:g}", **common)
        if -rope <= lo and hi <= rope:
            return RuleResult(Outcome.EQUIVALENT,
                              f"equivalent: interval [{lo:.3g}, {hi:.3g}] within ±{rope:g}", **common)

        needed = _needed_per_arm(per_arm_sd, rope, per_check_alpha, self.power, paired=ctx.paired)
        have = len(_paired_diffs(a, b)) if ctx.paired else min(len(a), len(b))
        more = max(0, needed - have) if needed is not None else None
        why = f"interval [{lo:.3g}, {hi:.3g}] overlaps ±{rope:g}"
        if ctx.is_last:
            why += f"; last checkpoint ({ctx.checkpoint}/{ctx.max_checkpoints}) reached"
        return RuleResult(undecided, why, needed_n=more, **common)


def _paired_diffs(a: Sample, b: Sample) -> list[float]:
    a_by_pair = {p: v for p, v in zip(a.pair_ids, a.values) if p is not None}
    return [v - a_by_pair[p] for p, v in zip(b.pair_ids, b.values) if p in a_by_pair]


def _needed_per_arm(sd: float, delta: float, alpha: float, power: float,
                    paired: bool) -> Optional[int]:
    """Normal-approximation sample size to detect a true difference of delta."""
    if sd <= 0:
        return None
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0) + NormalDist().inv_cdf(power)
    factor = 1.0 if paired else 2.0
    return math.ceil(factor * (z * sd / delta) ** 2)
