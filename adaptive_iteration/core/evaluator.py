"""core/evaluator.py — When to judge an experiment, and which data counts.

Schedule: an experiment is judged once per `window` after it starts (checkpoint 1,
2, …, max_windows). Between checkpoints evaluate() answers INSUFFICIENT without
recording anything, so re-running it daily cannot "peek" a false winner into the
ledger. Each checkpoint's decision is recorded once and then reused.

Validity: an observation only counts if it is mature (observed at least `maturity`
after the unit was produced), has a value for the metric, the value is finite, and
it lies inside MetricSpec.valid_range. Everything excluded is counted by reason.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from .decision import (
    Decision,
    DecisionContext,
    DecisionRule,
    Outcome,
    Sample,
    WelchIntervalRule,
)
from .ledger import Ledger
from .metrics import MetricSpec, Observation


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass
class Evaluator:
    ledger: Ledger
    rule: DecisionRule = field(default_factory=WelchIntervalRule)
    window: timedelta = timedelta(days=7)
    maturity: timedelta = timedelta(hours=72)
    max_windows: int = 4

    def evaluate(self, experiment_id: str, spec: MetricSpec,
                 now: Optional[datetime] = None) -> Decision:
        now = now or datetime.now(timezone.utc)
        exp = self.ledger.experiment(experiment_id)
        if exp is None:
            raise KeyError(experiment_id)

        final = self.ledger.final_decision(experiment_id)
        if final is not None:
            return final

        a, b, excluded = self.samples(experiment_id, spec, exp.variant_a.label,
                                      exp.variant_b.label)

        if not exp.started:
            return self._unrecorded(experiment_id, spec, 0, a, b, excluded,
                                    "experiment not started", now)
        elapsed = now - _parse(exp.started)
        checkpoint = min(int(elapsed / self.window), self.max_windows)
        if checkpoint < 1:
            nxt = _parse(exp.started) + self.window
            return self._unrecorded(experiment_id, spec, 0, a, b, excluded,
                                    f"before first checkpoint ({nxt.isoformat()})", now)

        for d in self.ledger.decisions(experiment_id):
            if d.checkpoint == checkpoint:
                return d

        ctx = DecisionContext(experiment_id=experiment_id, paired=exp.mode == "paired",
                              checkpoint=checkpoint, max_checkpoints=self.max_windows)
        result = self.rule.decide(a, b, spec, ctx)
        decision = Decision(
            experiment_id=experiment_id, outcome=result.outcome, metric=spec.name,
            checkpoint=checkpoint, n_a=len(a), n_b=len(b), excluded=excluded,
            rule=self.rule.name,
            rule_params={**result.params, "window_days": self.window / timedelta(days=1),
                         "maturity_hours": self.maturity / timedelta(hours=1),
                         "max_windows": self.max_windows, "metric_spec": spec.to_dict()},
            reason=result.reason, decided_at=now.isoformat(), effect=result.effect,
            interval=result.interval, confidence=result.confidence, needed_n=result.needed_n,
        )
        self.ledger.record_decision(decision)
        return decision

    def samples(self, experiment_id: str, spec: MetricSpec, label_a: str, label_b: str
                ) -> tuple[Sample, Sample, dict[str, int]]:
        excluded = {"immature": 0, "missing": 0, "non_finite": 0, "out_of_range": 0,
                    "unknown_variant": 0}
        arms: dict[str, list[Observation]] = {label_a: [], label_b: []}
        lo, hi = spec.valid_range
        for obs in self.ledger.observations(experiment_id):
            if obs.variant not in arms:
                excluded["unknown_variant"] += 1
                continue
            if _parse(obs.observed_at) - _parse(obs.produced_at) < self.maturity:
                excluded["immature"] += 1
                continue
            value = obs.metrics.get(spec.name)
            if value is None:
                excluded["missing"] += 1
                continue
            if not math.isfinite(value):
                excluded["non_finite"] += 1
                continue
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                excluded["out_of_range"] += 1
                continue
            arms[obs.variant].append(obs)

        def to_sample(obs_list: list[Observation]) -> Sample:
            return Sample(values=tuple(float(o.metrics[spec.name]) for o in obs_list),
                          unit_ids=tuple(o.unit_id for o in obs_list),
                          pair_ids=tuple(o.pair_id for o in obs_list))

        return to_sample(arms[label_a]), to_sample(arms[label_b]), excluded

    def _unrecorded(self, experiment_id: str, spec: MetricSpec, checkpoint: int,
                    a: Sample, b: Sample, excluded: dict[str, int], reason: str,
                    now: datetime) -> Decision:
        return Decision(
            experiment_id=experiment_id, outcome=Outcome.INSUFFICIENT, metric=spec.name,
            checkpoint=checkpoint, n_a=len(a), n_b=len(b), excluded=excluded,
            rule=self.rule.name, rule_params={}, reason=reason, decided_at=now.isoformat(),
        )
