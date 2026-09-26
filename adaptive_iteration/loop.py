"""loop.py — Loop: the driver that makes iteration automatic.

A domain plugs in three things:

    collect(experiment)                 -> observations for the units produced so far
    apply(experiment, variant, decision)   put a verdict into effect in the pipeline
    proposer                            -> proposes what to test next

and calls tick() on a schedule (cron, or an agent). Each tick:

1. collects fresh observations for every running experiment,
2. judges them (verdicts only at checkpoints, as always),
3. puts closed verdicts into effect: the winner if one arm won, otherwise variant A
   (keep what you had). A B-win can be held for human approval (require_approval);
4. fills free slots with new experiments from the proposer, after the usual review
   plus a detectability screen based on the domain's own data. With
   require_start_approval they wait, holding their slot, until approve_start().

If the pipeline changes under a running experiment (new model, prompt or config
version), call restart(): it abandons the old one and starts it again, so only data
from after the change counts. abandon() drops an experiment without a verdict.

While producing units, the pipeline asks variant_for(unit_id, stratum) which variant
of each running experiment a unit gets, so assignment stays balanced and on record.

tick() is idempotent: verdicts are recorded once per checkpoint and each experiment is
applied once. A hook that raises is reported in the tick's errors; the rest of the
tick still runs. When adopting Loop on a domain with older closed experiments that
were already handled, mark them with ledger.mark_applied first.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional

from .core.assignment import assign
from .core.decision import Decision, Outcome
from .core.domain import DomainConfig, config_for, current_config
from .core.experiment import Experiment, Variant
from .core.hypothesis import HypothesisEngine, Proposer
from .core.ledger import Ledger
from .core.lifecycle import abandon, restart
from .core.metrics import Observation
from .core.registry import DuplicateDetector
from .core.screening import Screening

Collect = Callable[[Experiment], Iterable[Observation]]
Apply = Callable[[Experiment, str, Decision], None]


@dataclass(frozen=True)
class Assignment:
    experiment_id: str
    variable: str
    variant: Variant


@dataclass
class TickReport:
    at: str
    collected: dict[str, int] = field(default_factory=dict)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    applied: list[dict[str, Any]] = field(default_factory=list)
    awaiting_approval: list[dict[str, Any]] = field(default_factory=list)
    started: list[dict[str, Any]] = field(default_factory=list)
    awaiting_start: list[dict[str, Any]] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


class Loop:
    def __init__(self, ledger: Ledger, domain: str, *, collect: Collect, apply: Apply,
                 proposer: Proposer, max_concurrent: int = 1, proposals_per_tick: int = 3,
                 require_approval: bool = False, require_start_approval: bool = False,
                 duplicates: Optional[DuplicateDetector] = None) -> None:
        if current_config(ledger, domain) is None:
            raise ValueError(f"domain {domain!r} is not configured; save a DomainConfig first")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be at least 1")
        self.ledger = ledger
        self.domain = domain
        self.collect = collect
        self.apply = apply
        self.proposer = proposer
        self.max_concurrent = max_concurrent
        self.proposals_per_tick = proposals_per_tick
        self.require_approval = require_approval
        self.require_start_approval = require_start_approval
        self.duplicates = duplicates

    # ── State ───────────────────────────────────────────────────────────────────

    def running(self) -> list[Experiment]:
        return [e for e in self.ledger.experiments(self.domain)
                if e.started and self.ledger.is_open(e.id)]

    def not_started(self) -> list[Experiment]:
        """Accepted experiments waiting to start (they hold a slot)."""
        return [e for e in self.ledger.experiments(self.domain)
                if not e.started and self.ledger.is_open(e.id)]

    def unapplied(self) -> list[tuple[Experiment, Decision]]:
        out = []
        for e in self.ledger.experiments(self.domain):
            d = self.ledger.final_decision(e.id)
            if d is not None and self.ledger.applied(e.id) is None:
                out.append((e, d))
        return out

    def pending_approval(self) -> list[Experiment]:
        return [e for e, d in self.unapplied() if d.outcome is Outcome.B_BETTER]

    # ── For the pipeline ──────────────────────────────────────────────────────

    def variant_for(self, unit_id: str, stratum: Optional[str] = None) -> list[Assignment]:
        """The variant this unit gets in every running interleaved experiment."""
        out = []
        for exp in self.running():
            if exp.mode == "paired":
                continue
            label = assign(self.ledger, exp.id, unit_id, stratum)
            variant = exp.variant_a if label == exp.variant_a.label else exp.variant_b
            out.append(Assignment(exp.id, exp.variable, variant))
        return out

    # ── The tick ──────────────────────────────────────────────────────────────

    def tick(self, now: Optional[datetime] = None) -> TickReport:
        now = now or datetime.now(timezone.utc)
        report = TickReport(at=now.isoformat())

        for exp in self.running():
            cfg = config_for(self.ledger, exp)
            try:
                report.collected[exp.id] = self._collect(exp, cfg)
            except Exception as e:  # noqa: BLE001 - a hook failure must not stop the tick
                report.errors.append(f"collect({exp.id}): {e}")
            d = cfg.build_evaluator(self.ledger).evaluate(exp.id, cfg.metric, now=now)
            report.verdicts.append({"experiment_id": exp.id, "variable": exp.variable,
                                    "outcome": d.outcome.value, "checkpoint": d.checkpoint,
                                    "effect": d.effect, "interval": d.interval,
                                    "reason": d.reason})

        for exp, d in self.unapplied():
            if d.outcome is Outcome.B_BETTER and self.require_approval:
                report.awaiting_approval.append(self._summary(exp, d, exp.variant_b.label))
                continue
            self._apply(exp, d, report)

        free = self.max_concurrent - len(self.running()) - len(self.not_started())
        if free > 0:
            self._start_new(free, now, report)
        report.awaiting_start = [self._plan(e) for e in self.not_started()]
        return report

    def approve(self, experiment_id: str) -> None:
        """Put a held B-win into effect."""
        exp = self.ledger.experiment(experiment_id)
        d = self.ledger.final_decision(experiment_id) if exp else None
        if exp is None or d is None or d.outcome is not Outcome.B_BETTER:
            raise ValueError(f"{experiment_id} has no B-win waiting for approval")
        report = TickReport(at=datetime.now(timezone.utc).isoformat())
        self._apply(exp, d, report)
        if report.errors:
            raise RuntimeError(report.errors[0])

    def approve_start(self, experiment_id: str, at: Optional[datetime] = None) -> None:
        """Start an experiment that was held by require_start_approval."""
        exp = self._waiting(experiment_id)
        self.ledger.start_experiment(exp.id, at=(at or datetime.now(timezone.utc)).isoformat())

    def reject_start(self, experiment_id: str, reason: str = "rejected before start") -> None:
        """Drop an experiment that was held by require_start_approval."""
        abandon(self.ledger, self._waiting(experiment_id).id, reason)

    def abandon(self, experiment_id: str, reason: str) -> None:
        abandon(self.ledger, experiment_id, reason)

    def restart(self, experiment_id: str, reason: str,
                at: Optional[datetime] = None) -> Experiment:
        """Abandon a running experiment and start it again from *at* (default now)."""
        return restart(self.ledger, experiment_id, reason,
                       at=(at or datetime.now(timezone.utc)).isoformat())

    # ── Internals ─────────────────────────────────────────────────────────────

    def _waiting(self, experiment_id: str) -> Experiment:
        for exp in self.not_started():
            if exp.id == experiment_id:
                return exp
        raise ValueError(f"{experiment_id} is not waiting to start")

    @staticmethod
    def _plan(exp: Experiment) -> dict[str, Any]:
        return {"experiment_id": exp.id, "variable": exp.variable,
                "variant_a": exp.variant_a.label, "variant_b": exp.variant_b.label,
                "proposed_by": exp.proposed_by, "expected_effect": exp.expected_effect}

    def _collect(self, exp: Experiment, cfg: DomainConfig) -> int:
        """Record observations that are new, changed, or were read before maturing."""
        maturity = timedelta(hours=cfg.maturity_hours)
        last = {o.unit_id: o for o in self.ledger.observations(exp.id)}
        fresh = []
        for obs in self.collect(exp):
            prev = last.get(obs.unit_id)
            if prev is None or prev.metrics != obs.metrics:
                fresh.append(obs)
            elif (_parse(prev.observed_at) - _parse(prev.produced_at) < maturity
                  and _parse(obs.observed_at) > _parse(prev.observed_at)):
                fresh.append(obs)      # same values, but now read at a later (maturer) time
        self.ledger.record_observations(fresh)
        return len(fresh)

    def _apply(self, exp: Experiment, d: Decision, report: TickReport) -> None:
        variant = exp.variant_b.label if d.outcome is Outcome.B_BETTER else exp.variant_a.label
        try:
            self.apply(exp, variant, d)
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"apply({exp.id}): {e}")
            return
        self.ledger.mark_applied(exp.id, variant, d.outcome.value)
        report.applied.append(self._summary(exp, d, variant))

    @staticmethod
    def _summary(exp: Experiment, d: Decision, variant: str) -> dict[str, Any]:
        return {"experiment_id": exp.id, "variable": exp.variable, "outcome": d.outcome.value,
                "variant": variant, "effect": d.effect, "interval": d.interval}

    def _start_new(self, free: int, now: datetime, report: TickReport) -> None:
        cfg = current_config(self.ledger, self.domain)
        assert cfg is not None
        screening = Screening(max_windows=cfg.max_windows,
                              window=timedelta(days=cfg.window_days),
                              binary=cfg.rule == "proportion")
        engine = HypothesisEngine(self.ledger, self.proposer, self.duplicates, screening)
        try:
            reviewed = engine.generate(self.domain, cfg.metric, n=self.proposals_per_tick)
        except Exception as e:  # noqa: BLE001
            report.errors.append(f"proposer: {e}")
            return
        for r in reviewed:
            report.proposals.append({"variable": r.variable or r.proposal.variable,
                                     "status": r.status, "flags": list(r.flags),
                                     "reason": r.reason, "screening": r.screening})
            if free == 0 or not r.accepted:
                continue
            exp = engine.accept(r)
            if not self.require_start_approval:
                self.ledger.start_experiment(exp.id, at=now.isoformat())
                report.started.append(self._plan(exp))
            free -= 1
