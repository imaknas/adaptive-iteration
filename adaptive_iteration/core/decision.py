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
from typing import Any, Literal, Optional, Protocol

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
    """Valid, mature values for one arm. pair_ids and strata align with values."""
    values: tuple[float, ...]
    unit_ids: tuple[str, ...] = ()
    pair_ids: tuple[Optional[str], ...] = ()
    strata: tuple[Optional[str], ...] = ()

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


def _check_common(alpha: float, min_n: int, superiority: str) -> None:
    if min_n < 2:
        raise ValueError("min_n must be at least 2 (a variance needs two values)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if superiority not in ("significance", "margin"):
        raise ValueError("superiority must be 'significance' or 'margin'")


def _classify(mean: float, lo: float, hi: float, rope: float, superiority: str,
              ctx: DecisionContext, needed_more: Optional[int], confidence: float,
              params: dict[str, Any], note: str = "") -> RuleResult:
    """Turn an interval for the (B-is-better-oriented) effect into an Outcome.

    significance: a winner needs the interval to exclude 0 and |effect| ≥ min_effect
    margin:       a winner needs the whole interval beyond ±min_effect
    equivalent:   the whole interval inside ±min_effect
    """
    undecided = Outcome.NO_DETECTABLE_DIFF if ctx.is_last else Outcome.INSUFFICIENT
    common = dict(effect=mean, interval=(lo, hi), confidence=confidence, params=params)
    bar = rope if superiority == "margin" else 0.0
    big_enough = superiority == "margin" or abs(mean) >= rope
    iv = f"interval [{lo:.3g}, {hi:.3g}]"
    if lo > bar and big_enough:
        return RuleResult(Outcome.B_BETTER,
                          f"B better: effect {mean:+.3g}, {iv} above {bar:+g}{note}", **common)
    if hi < -bar and big_enough:
        return RuleResult(Outcome.A_BETTER,
                          f"A better: effect {mean:+.3g}, {iv} below {-bar:+g}{note}", **common)
    if -rope <= lo and hi <= rope:
        return RuleResult(Outcome.EQUIVALENT, f"equivalent: {iv} within ±{rope:g}{note}",
                          **common)
    why = f"{iv} overlaps ±{rope:g}{note}"
    if ctx.is_last:
        why += f"; last checkpoint ({ctx.checkpoint}/{ctx.max_checkpoints}) reached"
    return RuleResult(undecided, why, needed_n=needed_more, **common)


@dataclass(frozen=True)
class WelchIntervalRule:
    """Default rule: confidence interval for mean(B) − mean(A) compared with ±min_effect.

    - interleaved: Welch t interval (unequal variances)
    - paired: one-sample t interval on per-pair differences
    - stratified (units carry a stratum, e.g. topic): the effect is estimated within
      each stratum and averaged by stratum size, so an uneven mix of strata between
      the arms cannot masquerade as an effect. Strata missing one arm are left out.
      Uses a pooled within-stratum variance (assumes similar spread across strata).
    - alpha is Bonferroni-split over max_checkpoints, so checking every window
      keeps the experiment-wide false-positive rate at or below alpha.

    superiority decides when one arm "wins":
    - "significance" (default): the interval excludes 0 and the point estimate is at
      least min_effect — confidently better, by an amount that looks meaningful
    - "margin": the whole interval clears min_effect — confidently better by at least
      min_effect. Stricter; a true effect equal to min_effect is never declared.
    """
    alpha: float = 0.05
    min_n: int = 5
    power: float = 0.8
    superiority: Literal["significance", "margin"] = "significance"
    stratify: bool = True
    name: str = "welch_interval"

    def __post_init__(self) -> None:
        _check_common(self.alpha, self.min_n, self.superiority)

    def decide(self, a: Sample, b: Sample, spec: MetricSpec, ctx: DecisionContext) -> RuleResult:
        per_check_alpha = self.alpha / ctx.max_checkpoints
        confidence = 1.0 - per_check_alpha
        params = {"alpha": self.alpha, "min_n": self.min_n, "power": self.power,
                  "superiority": self.superiority, "per_check_alpha": per_check_alpha,
                  "min_effect": spec.min_effect, "stratified": False}
        sign = 1.0 if spec.higher_is_better else -1.0
        undecided = Outcome.NO_DETECTABLE_DIFF if ctx.is_last else Outcome.INSUFFICIENT
        note = ""

        def too_few(msg: str, have: int) -> RuleResult:
            return RuleResult(undecided, msg, confidence=confidence, params=params,
                              needed_n=self.min_n - have)

        if ctx.paired:
            diffs = _paired_diffs(a, b)
            n = len(diffs)
            if n < self.min_n:
                return too_few(f"{n} complete pairs < min_n={self.min_n}", n)
            mean = statistics.fmean(diffs) * sign
            sd = statistics.stdev(diffs)
            se, df, per_arm_sd, have = sd / math.sqrt(n), n - 1, sd, n
        elif self.stratify and any(s is not None for s in (*a.strata, *b.strata)):
            est = _stratified(a, b)
            params["stratified"] = True
            params["strata_used"] = est.strata_used
            if est.dropped:
                note = f"; {est.dropped} units in strata without both arms left out"
                params["dropped_units"] = est.dropped
            if min(est.n_a, est.n_b) < self.min_n or est.df < 1:
                return too_few(f"n_a={est.n_a}, n_b={est.n_b} in strata with both arms; "
                               f"each arm needs ≥ min_n={self.min_n}{note}",
                               min(est.n_a, est.n_b))
            mean, se, df = est.effect * sign, est.se, est.df
            per_arm_sd, have = est.sd, min(est.n_a, est.n_b)
        else:
            n_a, n_b = len(a), len(b)
            if min(n_a, n_b) < self.min_n:
                return too_few(f"n_a={n_a}, n_b={n_b}; each arm needs ≥ min_n={self.min_n}",
                               min(n_a, n_b))
            var_a, var_b = statistics.variance(a.values), statistics.variance(b.values)
            mean = (statistics.fmean(b.values) - statistics.fmean(a.values)) * sign
            se2 = var_a / n_a + var_b / n_b
            se = math.sqrt(se2)
            if se2 > 0:
                df = se2 ** 2 / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
            else:
                df = n_a + n_b - 2
            per_arm_sd, have = math.sqrt((var_a + var_b) / 2.0), min(n_a, n_b)

        if se == 0:
            # No variation at all (e.g. a 0/1 metric with no successes in either arm):
            # the interval collapses to a point and would claim certainty it doesn't have.
            return RuleResult(undecided, "no variation in the data; interval not estimable",
                              effect=mean, confidence=confidence, params=params)

        half = t_ppf(1.0 - per_check_alpha / 2.0, df) * se
        needed = _needed_per_arm(per_arm_sd, spec.min_effect, per_check_alpha, self.power,
                                 paired=ctx.paired)
        more = max(0, needed - have) if needed is not None else None
        return _classify(mean, mean - half, mean + half, spec.min_effect, self.superiority,
                         ctx, more, confidence, params, note)


@dataclass(frozen=True)
class ProportionIntervalRule:
    """For 0/1 metrics (replied, clicked, converted): Newcombe's hybrid score interval
    for p(B) − p(A), built from per-arm Wilson intervals.

    Unlike a t interval it stays honest when events are rare — zero successes in both
    arms gives a wide interval, not a collapsed one. Outcomes, superiority and the
    Bonferroni split work exactly as in WelchIntervalRule. min_effect is in proportion
    units (0.10 = ten percentage points). Interleaved experiments only; strata ignored.
    """
    alpha: float = 0.05
    min_n: int = 5
    power: float = 0.8
    superiority: Literal["significance", "margin"] = "significance"
    name: str = "proportion_interval"

    def __post_init__(self) -> None:
        _check_common(self.alpha, self.min_n, self.superiority)

    def decide(self, a: Sample, b: Sample, spec: MetricSpec, ctx: DecisionContext) -> RuleResult:
        if ctx.paired:
            raise ValueError("ProportionIntervalRule does not support paired experiments")
        for v in (*a.values, *b.values):
            if v not in (0.0, 1.0):
                raise ValueError(f"ProportionIntervalRule needs 0/1 values, got {v!r}")
        per_check_alpha = self.alpha / ctx.max_checkpoints
        confidence = 1.0 - per_check_alpha
        params = {"alpha": self.alpha, "min_n": self.min_n, "power": self.power,
                  "superiority": self.superiority, "per_check_alpha": per_check_alpha,
                  "min_effect": spec.min_effect}
        n_a, n_b = len(a), len(b)
        if min(n_a, n_b) < self.min_n:
            undecided = Outcome.NO_DETECTABLE_DIFF if ctx.is_last else Outcome.INSUFFICIENT
            return RuleResult(undecided, f"n_a={n_a}, n_b={n_b}; each arm needs ≥ "
                              f"min_n={self.min_n}", confidence=confidence, params=params,
                              needed_n=self.min_n - min(n_a, n_b))

        z = NormalDist().inv_cdf(1.0 - per_check_alpha / 2.0)
        p_a, p_b = sum(a.values) / n_a, sum(b.values) / n_b
        l_a, u_a = _wilson(p_a, n_a, z)
        l_b, u_b = _wilson(p_b, n_b, z)
        d = p_b - p_a
        lo = d - math.sqrt((p_b - l_b) ** 2 + (u_a - p_a) ** 2)
        hi = d + math.sqrt((u_b - p_b) ** 2 + (p_a - l_a) ** 2)
        if not spec.higher_is_better:
            d, lo, hi = -d, -hi, -lo
        params.update({"p_a": p_a, "p_b": p_b})

        pooled = (sum(a.values) + sum(b.values)) / (n_a + n_b)
        needed = _needed_per_arm(math.sqrt(pooled * (1 - pooled)), spec.min_effect,
                                 per_check_alpha, self.power, paired=False)
        more = max(0, needed - min(n_a, n_b)) if needed is not None else None
        note = f" (p_a={p_a:.3g}, p_b={p_b:.3g})"
        return _classify(d, lo, hi, spec.min_effect, self.superiority, ctx, more,
                         confidence, params, note)


def _wilson(p: float, n: int, z: float) -> tuple[float, float]:
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class _StratifiedEstimate:
    effect: float
    se: float
    df: int
    sd: float
    n_a: int
    n_b: int
    strata_used: int
    dropped: int


def _stratified(a: Sample, b: Sample) -> _StratifiedEstimate:
    """Post-stratified difference in means with a pooled within-cell variance."""
    def by_stratum(s: Sample) -> dict[Optional[str], list[float]]:
        groups: dict[Optional[str], list[float]] = {}
        strata = s.strata or tuple(None for _ in s.values)
        for st, v in zip(strata, s.values):
            groups.setdefault(st, []).append(v)
        return groups

    ga, gb = by_stratum(a), by_stratum(b)
    shared = [st for st in ga if st in gb]
    dropped = (sum(len(v) for st, v in ga.items() if st not in gb)
               + sum(len(v) for st, v in gb.items() if st not in ga))
    n_a = sum(len(ga[st]) for st in shared)
    n_b = sum(len(gb[st]) for st in shared)
    total = n_a + n_b
    df = total - 2 * len(shared)
    if not shared or df < 1:
        return _StratifiedEstimate(0.0, 0.0, max(df, 0), 0.0, n_a, n_b, len(shared), dropped)

    ss = 0.0
    effect = 0.0
    var_factor = 0.0
    for st in shared:
        xa, xb = ga[st], gb[st]
        ma, mb = statistics.fmean(xa), statistics.fmean(xb)
        ss += sum((x - ma) ** 2 for x in xa) + sum((x - mb) ** 2 for x in xb)
        w = (len(xa) + len(xb)) / total
        effect += w * (mb - ma)
        var_factor += w * w * (1 / len(xa) + 1 / len(xb))
    sigma2 = ss / df
    return _StratifiedEstimate(effect, math.sqrt(sigma2 * var_factor), df, math.sqrt(sigma2),
                               n_a, n_b, len(shared), dropped)


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
