"""replay.py — Judge the judge: evaluate decision rules against recorded data.

Adapted from the replay idea in Dream-RSI (arXiv 2609.14858): recorded history is
reused as a simulator, so a candidate rule can be compared with the incumbent
cheaply, and it is only adopted if it is not worse. As there, replay never invents
outcomes that were not observed — it can evaluate *how to judge and schedule*
experiments, not *which untested hypothesis* would win.

Two tools:

calibrate()   Resamples a pool of real per-unit values (e.g. every video's metric)
              into simulated experiments. effect=0 measures the false-positive rate
              under your real noise; effect>0 measures power and time-to-decision.
replay()      Re-runs the Evaluator week by week over an experiment's recorded
              observations, as if the rule had been live from the start.

gate() combines calibrate() runs into an adopt / keep-incumbent verdict.
"""
from __future__ import annotations

import random
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional, Sequence

from .core.decision import Decision, DecisionContext, DecisionRule, Outcome, Sample
from .core.evaluator import Evaluator
from .core.experiment import Experiment
from .core.ledger import Ledger
from .core.metrics import MetricSpec, Observation


@dataclass(frozen=True)
class Calibration:
    rule: str
    effect: float
    sims: int
    outcomes: dict[str, int]
    median_weeks_to_final: Optional[float]
    mean_units_per_arm: float

    @property
    def wrong_direction(self) -> float:
        """Share of runs that declared the worse arm the winner (A_BETTER when B is
        truly ahead, or any winner when there is no true effect)."""
        a = self.outcomes.get(Outcome.A_BETTER.value, 0)
        b = self.outcomes.get(Outcome.B_BETTER.value, 0)
        wrong = a + b if self.effect == 0 else (a if self.effect > 0 else b)
        return wrong / self.sims

    @property
    def detected(self) -> float:
        """Share of runs that declared the truly better arm (0 when effect == 0)."""
        if self.effect == 0:
            return 0.0
        key = Outcome.B_BETTER if self.effect > 0 else Outcome.A_BETTER
        return self.outcomes.get(key.value, 0) / self.sims


def calibrate(pool: Sequence[float], rule: DecisionRule, spec: MetricSpec, *,
              effect: float = 0.0, per_window: int, windows: int = 4, sims: int = 1000,
              seed: int = 0, pool_b: Optional[Sequence[float]] = None) -> Calibration:
    """Simulate experiments by drawing both arms from *pool* (with replacement),
    shifting arm B by *effect*, and asking *rule* at every weekly checkpoint.

    effect is in "B is better" units: it is added to B when higher_is_better,
    subtracted otherwise.

    For metrics a shift would break (0/1 outcomes), pass *pool_b* instead: B is drawn
    from it unshifted, and *effect* is only the label used to score the result
    (the true B − A difference, e.g. 0.10 for 20% vs 30%).
    """
    if len(pool) < 2 or (pool_b is not None and len(pool_b) < 2):
        raise ValueError("pools need at least two values")
    rng = random.Random(seed)
    shift = effect if spec.higher_is_better else -effect
    counts: Counter[str] = Counter()
    weeks: list[int] = []
    units: list[int] = []
    for _ in range(sims):
        a: list[float] = []
        b: list[float] = []
        outcome = Outcome.INSUFFICIENT
        for k in range(1, windows + 1):
            a += rng.choices(pool, k=per_window)
            if pool_b is None:
                b += [v + shift for v in rng.choices(pool, k=per_window)]
            else:
                b += rng.choices(pool_b, k=per_window)
            ctx = DecisionContext("calibration", paired=False, checkpoint=k,
                                  max_checkpoints=windows)
            outcome = rule.decide(Sample(tuple(a)), Sample(tuple(b)), spec, ctx).outcome
            if outcome.is_final:
                weeks.append(k)
                break
        counts[outcome.value] += 1
        units.append(len(a))
    return Calibration(rule=_rule_label(rule), effect=effect, sims=sims, outcomes=dict(counts),
                       median_weeks_to_final=statistics.median(weeks) if weeks else None,
                       mean_units_per_arm=statistics.fmean(units))


@dataclass(frozen=True)
class GateResult:
    adopt: bool
    reasons: tuple[str, ...]
    candidate: tuple[Calibration, ...]
    incumbent: tuple[Calibration, ...]


def gate(candidate: DecisionRule, incumbent: DecisionRule, pool: Sequence[float],
         spec: MetricSpec, *, effects: Iterable[float], per_window: int, windows: int = 4,
         sims: int = 1000, max_false_positive: float = 0.05, seed: int = 0) -> GateResult:
    """Adopt *candidate* only if, on the same resampled data, its false-positive rate
    stays within *max_false_positive* and it detects real effects at least as often
    as *incumbent* (within Monte Carlo noise) for every effect size in *effects*."""
    all_effects = [0.0, *[e for e in effects if e != 0]]
    cand = tuple(calibrate(pool, candidate, spec, effect=e, per_window=per_window,
                           windows=windows, sims=sims, seed=seed) for e in all_effects)
    inc = tuple(calibrate(pool, incumbent, spec, effect=e, per_window=per_window,
                          windows=windows, sims=sims, seed=seed) for e in all_effects)
    reasons = []
    tolerance = 2.0 * (0.25 / sims) ** 0.5   # ~2 standard errors of a proportion
    if cand[0].wrong_direction > max_false_positive:
        reasons.append(f"false-positive rate {cand[0].wrong_direction:.1%} "
                       f"> {max_false_positive:.0%}")
    for c, i in zip(cand[1:], inc[1:]):
        if c.detected + tolerance < i.detected:
            reasons.append(f"effect {c.effect:g}: detects {c.detected:.1%} "
                           f"vs incumbent {i.detected:.1%}")
        if c.wrong_direction > max(i.wrong_direction, 0.0) + tolerance:
            reasons.append(f"effect {c.effect:g}: wrong-direction {c.wrong_direction:.1%} "
                           f"vs incumbent {i.wrong_direction:.1%}")
    return GateResult(adopt=not reasons, reasons=tuple(reasons), candidate=cand, incumbent=inc)


def replay(experiment: Experiment, observations: Sequence[Observation], spec: MetricSpec, *,
           rule: Optional[DecisionRule] = None, window: timedelta = timedelta(days=7),
           maturity: timedelta = timedelta(hours=72), max_windows: int = 4
           ) -> list[Decision]:
    """Decisions the Evaluator would have recorded at each weekly checkpoint.

    Only observations whose unit was produced early enough to be mature at a
    checkpoint are visible there, so later units never leak into earlier weeks.
    Metric values are whatever was recorded: if they were read after the checkpoint,
    replay uses those later values (the best data available), not a historical snapshot.
    """
    if not experiment.started:
        raise ValueError("experiment.started is required for replay")
    start = _parse(experiment.started)
    decisions = []
    for k in range(1, max_windows + 1):
        at = start + k * window
        ledger = Ledger(None)
        exp = Experiment.from_dict(experiment.to_dict())
        ledger.add_experiment(exp)
        ledger.start_experiment(exp.id, at=experiment.started)
        for obs in observations:
            produced = _parse(obs.produced_at)
            if produced + maturity > at:
                continue
            ledger.record_observation(Observation(
                experiment_id=exp.id, variant=obs.variant, unit_id=obs.unit_id,
                produced_at=obs.produced_at,
                observed_at=max(produced + maturity, min(at, _parse(obs.observed_at))
                                ).isoformat(),
                metrics=obs.metrics, pair_id=obs.pair_id))
        kwargs = {"rule": rule} if rule is not None else {}
        evaluator = Evaluator(ledger, window=window, maturity=maturity,
                              max_windows=max_windows, **kwargs)
        d = evaluator.evaluate(exp.id, spec, now=at)
        decisions.append(d)
        if d.outcome.is_final:
            break
    return decisions


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _rule_label(rule: DecisionRule) -> str:
    extra = getattr(rule, "superiority", None)
    return f"{rule.name}[{extra}]" if extra else rule.name


__all__ = ["Calibration", "GateResult", "calibrate", "gate", "replay"]
