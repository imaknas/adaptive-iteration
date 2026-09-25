"""core/evidence.py — EvidenceSummary: what the ledger currently supports, as plain data.

This is the input a Proposer receives. It contains no prompt text; to_markdown() is
only a convenience for users who want to feed it to a language model themselves.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal, Optional

from .decision import Outcome
from .ledger import Ledger
from .metrics import MetricSpec
from .registry import VariableDef, VariableRegistry

Status = Literal["untested", "open", "concluded", "equivalent", "no_detectable_diff",
                 "legacy_unverified"]


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


@dataclass(frozen=True)
class EvidenceSummary:
    domain: str
    metric: MetricSpec
    variables: tuple[VariableEvidence, ...]
    open_experiments: tuple[str, ...]
    metric_dispersion: dict[str, float]
    data_quality: dict[str, int]

    def to_markdown(self) -> str:
        lines = [f"# Evidence — {self.domain}",
                 f"Metric: {self.metric.name} "
                 f"({'higher' if self.metric.higher_is_better else 'lower'} is better, "
                 f"min meaningful effect {self.metric.min_effect:g})", ""]
        lines.append("| variable | status | best | effect [interval] | n |")
        lines.append("|---|---|---|---|---|")
        for v in self.variables:
            eff = "" if v.effect is None else f"{v.effect:+.3g}"
            if v.interval:
                eff += f" [{v.interval[0]:.3g}, {v.interval[1]:.3g}]"
            lines.append(f"| {v.variable} | {v.status} | {v.best_variant or ''} | {eff} "
                         f"| {v.n_observations} |")
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

    names = set(by_var) | set(legacy_by_var) | {d.name for d in registry.variables()}
    open_ids: list[str] = []
    evidence: list[VariableEvidence] = []
    for name in sorted(n for n in names if n):
        exps = by_var.get(name, [])
        n_obs = sum(len(ledger.observations(e.id)) for e in exps)
        still_open = [e.id for e in exps if ledger.final_decision(e.id) is None]
        open_ids += still_open
        finals = [(ledger.final_decision(e.id), e) for e in exps]
        finals = [(d, e) for d, e in finals if d is not None]
        status: Status
        best = effect = interval = None
        if still_open:
            status = "open"
        elif finals:
            d, e = max(finals, key=lambda de: de[0].decided_at)
            status = _STATUS_BY_OUTCOME[d.outcome]
            effect, interval = d.effect, d.interval
            if d.outcome is Outcome.B_BETTER:
                best = e.variant_b.label
            elif d.outcome is Outcome.A_BETTER:
                best = e.variant_a.label
        elif legacy_by_var.get(name):
            status = "legacy_unverified"
            n_obs = legacy_by_var[name]
        else:
            status = "untested"
        evidence.append(VariableEvidence(
            variable=name, status=status, best_variant=best, effect=effect, interval=interval,
            n_observations=n_obs, experiments=tuple(e.id for e in exps),
            definition=registry.get(name),
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

    return EvidenceSummary(domain=domain, metric=spec, variables=tuple(evidence),
                           open_experiments=tuple(open_ids), metric_dispersion=dispersion,
                           data_quality=quality)
