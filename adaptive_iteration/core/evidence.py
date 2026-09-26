"""core/evidence.py — EvidenceSummary: what the ledger currently supports, as plain data.

This is the input a Proposer receives. It contains no prompt text; to_markdown() is
only a convenience for users who want to feed it to a language model themselves.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Literal, Optional

from .decision import Outcome
from .ledger import Ledger
from .metrics import MetricSpec
from .registry import VariableDef, VariableRegistry
from .shrinkage import Prior, approx_se, estimate_prior

Status = Literal["untested", "open", "concluded", "equivalent", "no_detectable_diff",
                 "abandoned", "legacy_unverified"]


@dataclass(frozen=True)
class VariableEvidence:
    variable: str
    status: Status
    best_variant: Optional[str]
    effect: Optional[float]
    interval: Optional[tuple[float, float]]
    n_observations: int
    experiments: tuple[str, ...]
    definition: Optional[VariableDef] = None
    needed_n: Optional[int] = None   # open experiments: latest estimate of units still needed per arm
    shrunk_effect: Optional[float] = None  # effect corrected for the winner's curse (shrinkage.py)


@dataclass(frozen=True)
class EvidenceSummary:
    domain: str
    metric: MetricSpec
    variables: tuple[VariableEvidence, ...]
    open_experiments: tuple[str, ...]
    metric_dispersion: dict[str, float]
    data_quality: dict[str, int]
    cost: dict[str, Optional[float]]   # what judging has cost so far, see build_evidence()
    proposers: dict[str, dict[str, Optional[float]]] = field(default_factory=dict)
    shrinkage: dict[str, object] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [f"# Evidence — {self.domain}",
                 f"Metric: {self.metric.name} "
                 f"({'higher' if self.metric.higher_is_better else 'lower'} is better, "
                 f"min meaningful effect {self.metric.min_effect:g})", ""]
        lines.append("| variable | status | best | effect [interval] | n | still needed/arm |")
        lines.append("|---|---|---|---|---|---|")
        for v in self.variables:
            eff = "" if v.effect is None else f"{v.effect:+.3g}"
            if v.interval:
                eff += f" [{v.interval[0]:.3g}, {v.interval[1]:.3g}]"
            if v.shrunk_effect is not None:
                eff += f" → {v.shrunk_effect:+.3g} corrected"
            need = "" if v.needed_n is None else str(v.needed_n)
            lines.append(f"| {v.variable} | {v.status} | {v.best_variant or ''} | {eff} "
                         f"| {v.n_observations} | {need} |")
        defs = [v.definition for v in self.variables if v.definition]
        if defs:
            lines += ["", "## Registered variables"]
            for d in defs:
                extra = f" — aliases: {', '.join(d.aliases)}" if d.aliases else ""
                how = f" — execution: {d.execution}" if d.execution else ""
                lines.append(f"- `{d.name}`: {d.description}{extra}{how}")
        if self.metric_dispersion:
            lines += ["", "## Metric dispersion (stdev across units)"]
            lines += [f"- {k}: {v:.3g}" for k, v in sorted(self.metric_dispersion.items())]
        if self.cost.get("closed_experiments"):
            c = self.cost
            per = c["units_per_decisive"]
            lines += ["", "## Cost of judging",
                      f"- closed experiments: {c['closed_experiments']:g}, decisive: "
                      f"{c['decisive']:g}, no detectable difference: {c['no_detectable_diff']:g}",
                      f"- units per decisive result: {per:.0f}" if per is not None
                      else "- units per decisive result: (no decisive result yet)"]
        if self.shrinkage:
            lines += ["", "## Winner's-curse correction",
                      f"- {self.shrinkage['basis']}",
                      "- \"corrected\" effects pull each measured effect toward 0 by how noisy "
                      "it is; verdicts are unchanged"]
        if self.proposers:
            lines += ["", "## Proposer track record",
                      "| proposer | proposed | closed | challenger won | control won | "
                      "no difference | units per decisive | realised ÷ expected effect |",
                      "|---|---|---|---|---|---|---|---|"]
            for name, r in sorted(self.proposers.items()):
                per = "" if r["units_per_decisive"] is None else f"{r['units_per_decisive']:.0f}"
                ratio = "" if r["effect_ratio"] is None else f"{r['effect_ratio']:.2f}"
                lines.append(f"| {name} | {r['proposed']:g} | {r['closed']:g} | "
                             f"{r['challenger_won']:g} | {r['control_won']:g} | "
                             f"{r['no_difference']:g} | {per} | {ratio} |")
        if any(self.data_quality.values()):
            lines += ["", "## Data quality",
                      *[f"- {k}: {v}" for k, v in self.data_quality.items() if v]]
        return "\n".join(lines)


_STATUS_BY_OUTCOME: dict[Outcome, Status] = {
    Outcome.A_BETTER: "concluded",
    Outcome.B_BETTER: "concluded",
    Outcome.EQUIVALENT: "equivalent",
    Outcome.NO_DETECTABLE_DIFF: "no_detectable_diff",
}


def build_evidence(ledger: Ledger, domain: str, spec: MetricSpec) -> EvidenceSummary:
    registry = VariableRegistry(ledger, domain)
    canonical = lambda name: registry.resolve(name) or name  # noqa: E731

    experiments = ledger.experiments(domain=domain)
    by_var: dict[str, list] = {}
    for exp in experiments:
        by_var.setdefault(canonical(exp.variable), []).append(exp)
    legacy_by_var: dict[str, int] = {}
    for r in ledger.legacy_records(domain):
        v = canonical(r.get("variable") or "")
        legacy_by_var[v] = legacy_by_var.get(v, 0) + 1

    prior = estimate_prior(d for d in (ledger.final_decision(e.id) for e in experiments)
                           if d is not None)
    names = set(by_var) | set(legacy_by_var) | {d.name for d in registry.variables()}
    open_ids: list[str] = []
    evidence: list[VariableEvidence] = []
    for name in sorted(n for n in names if n):
        exps = by_var.get(name, [])
        n_obs = sum(len(ledger.observations(e.id)) for e in exps)
        still_open = [e.id for e in exps if ledger.is_open(e.id)]
        open_ids += still_open
        finals = [(ledger.final_decision(e.id), e) for e in exps]
        finals = [(d, e) for d, e in finals if d is not None]
        status: Status
        best = effect = interval = needed = shrunk = None
        if still_open:
            status = "open"
            latest = [d for e_id in still_open for d in ledger.decisions(e_id)]
            if latest:
                needed = max(latest, key=lambda d: d.decided_at).needed_n
        elif finals:
            d, e = max(finals, key=lambda de: de[0].decided_at)
            status = _STATUS_BY_OUTCOME[d.outcome]
            effect, interval = d.effect, d.interval
            shrunk = prior.shrink(d.effect, approx_se(d))
            if d.outcome is Outcome.B_BETTER:
                best = e.variant_b.label
            elif d.outcome is Outcome.A_BETTER:
                best = e.variant_a.label
        elif exps:
            status = "abandoned"            # every experiment on it was abandoned
        elif legacy_by_var.get(name):
            status = "legacy_unverified"
            n_obs = legacy_by_var[name]
        else:
            status = "untested"
        evidence.append(VariableEvidence(
            variable=name, status=status, best_variant=best, effect=effect, interval=interval,
            n_observations=n_obs, experiments=tuple(e.id for e in exps),
            definition=registry.get(name), needed_n=needed, shrunk_effect=shrunk,
        ))

    values: dict[str, list[float]] = {}
    quality = {"missing": 0, "non_finite": 0}
    for exp in experiments:
        for obs in ledger.observations(exp.id):
            if obs.metrics.get(spec.name) is None:
                quality["missing"] += 1
            for k, v in obs.metrics.items():
                if v is None:
                    continue
                if not math.isfinite(v):
                    if k == spec.name:
                        quality["non_finite"] += 1
                    continue
                values.setdefault(k, []).append(float(v))
    dispersion = {k: statistics.stdev(v) for k, v in values.items() if len(v) >= 2}

    # Cost: units consumed by closed experiments per decisive (A/B better) result, so a
    # proposer can favour hypotheses that resolve quickly over ones that run out the clock.
    closed = [(e, ledger.final_decision(e.id)) for e in experiments]
    closed = [(e, d) for e, d in closed if d is not None]
    decisive = sum(d.outcome in (Outcome.A_BETTER, Outcome.B_BETTER) for _, d in closed)
    units = sum(len(ledger.observations(e.id)) for e, _ in closed)
    cost = {
        "closed_experiments": float(len(closed)),
        "decisive": float(decisive),
        "no_detectable_diff": float(sum(d.outcome is Outcome.NO_DETECTABLE_DIFF
                                        for _, d in closed)),
        "units": float(units),
        "units_per_decisive": units / decisive if decisive else None,
    }

    return EvidenceSummary(domain=domain, metric=spec, variables=tuple(evidence),
                           open_experiments=tuple(open_ids), metric_dispersion=dispersion,
                           data_quality=quality, cost=cost,
                           proposers=_track_record(ledger, experiments, prior),
                           shrinkage={"tau": prior.tau, "n_experiments": prior.n_experiments,
                                      "basis": prior.basis})


def _track_record(ledger: Ledger, experiments: list,
                  prior: Prior) -> dict[str, dict[str, Optional[float]]]:
    """Per proposer: how its experiments ended, what they cost, and how its expected
    effects compared with what was measured (median of realised ÷ expected)."""
    groups: dict[str, list] = {}
    for exp in experiments:
        groups.setdefault(exp.proposed_by or "unknown", []).append(exp)
    out: dict[str, dict[str, Optional[float]]] = {}
    for name, exps in groups.items():
        finals = [(e, ledger.final_decision(e.id)) for e in exps]
        closed = [(e, d) for e, d in finals if d is not None]
        abandoned = sum(ledger.abandoned(e.id) is not None for e in exps)
        b_won = sum(d.outcome is Outcome.B_BETTER for _, d in closed)
        a_won = sum(d.outcome is Outcome.A_BETTER for _, d in closed)
        units = sum(len(ledger.observations(e.id)) for e, _ in closed)
        ratios = [d.effect / e.expected_effect for e, d in closed
                  if e.expected_effect and d.effect is not None]
        shrunk = [(prior.shrink(d.effect, approx_se(d)), e.expected_effect) for e, d in closed
                  if e.expected_effect and d.effect is not None]
        shrunk_ratios = [s / x for s, x in shrunk if s is not None]
        out[name] = {
            "proposed": float(len(exps)), "closed": float(len(closed)),
            "challenger_won": float(b_won), "control_won": float(a_won),
            "no_difference": float(len(closed) - b_won - a_won),
            "abandoned": float(abandoned),
            "units_per_decisive": units / (a_won + b_won) if a_won + b_won else None,
            "effect_ratio": statistics.median(ratios) if ratios else None,
            "corrected_effect_ratio": (statistics.median(shrunk_ratios)
                                       if shrunk_ratios else None),
        }
    return out
