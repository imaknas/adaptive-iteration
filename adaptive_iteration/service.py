"""service.py — The framework as plain operations on a ledger file.

Both the CLI (cli.py) and the MCP server (mcp_server.py) are thin wrappers over
these functions, so an agent gets identical behaviour either way. Every function
takes a ledger path, returns JSON-serialisable data, and raises ServiceError with a
message that says what to do next.

Typical loop:
    configure → register_variable → review_proposals / accept_proposal
    → record_observations (repeatedly) → evaluate (daily is fine) → evidence → …
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Union

from .core.assignment import assign as _assign
from .core.decision import Decision, Outcome
from .core.domain import DomainConfig, config_for, current_config, save_config
from .core.evidence import build_evidence
from .core.experiment import Experiment, Variant
from .core.hypothesis import HypothesisEngine, Proposal, ReviewedProposal
from .core.ledger import Ledger
from .core.lifecycle import abandon as _abandon
from .core.lifecycle import restart as _restart
from .core.metrics import MetricSpec, Observation
from .core.registry import VariableDef, VariableRegistry
from .core.screening import Screening
from .core.shrinkage import approx_se, estimate_prior
from .replay import calibrate as _calibrate

PathLike = Union[str, Path]


class ServiceError(ValueError):
    """A request that cannot be carried out; the message says how to fix it."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _open(ledger: PathLike) -> Ledger:
    led = Ledger(Path(ledger))
    if led.read_only:
        raise ServiceError(f"{ledger} is a v0.1 ledger; convert it first with "
                           "adaptive_iteration.migrate.v1_to_v2(src, dst)")
    return led


def _config(led: Ledger, domain: str) -> DomainConfig:
    cfg = current_config(led, domain)
    if cfg is None:
        raise ServiceError(f"domain {domain!r} is not configured; call configure first "
                           "(metric name, min_effect, rule)")
    return cfg


# ── Setup ───────────────────────────────────────────────────────────────────────

def configure(ledger: PathLike, domain: str, metric: str, min_effect: float, *,
              higher_is_better: bool = True, rule: str = "welch",
              superiority: str = "significance", window_days: float = 7.0,
              maturity_hours: float = 72.0, max_windows: int = 4,
              valid_min: Optional[float] = None, valid_max: Optional[float] = None
              ) -> dict[str, Any]:
    """Set how a domain's experiments are judged. Applies to experiments started from
    now on; running experiments keep the settings they started with."""
    led = _open(ledger)
    try:
        cfg = DomainConfig(domain=domain,
                           metric=MetricSpec(name=metric, min_effect=min_effect,
                                             higher_is_better=higher_is_better,
                                             valid_range=(valid_min, valid_max)),
                           rule=rule, superiority=superiority, window_days=window_days,
                           maturity_hours=maturity_hours, max_windows=max_windows)
    except ValueError as e:
        raise ServiceError(str(e)) from e
    save_config(led, cfg)
    running = [e.id for e in led.experiments(domain) if led.is_open(e.id)
               and e.started]
    return {"config": cfg.to_dict(),
            "note": ("applies to experiments started from now on"
                     + (f"; {len(running)} running experiment(s) keep their original settings"
                        if running else ""))}


def register_variable(ledger: PathLike, domain: str, name: str, description: str = "",
                      execution: Optional[str] = None) -> dict[str, Any]:
    led = _open(ledger)
    try:
        VariableRegistry(led, domain).register(VariableDef(name, description, (), execution))
    except ValueError as e:
        raise ServiceError(str(e)) from e
    return {"registered": name, "domain": domain}


def merge_variables(ledger: PathLike, domain: str, name: str, into: str) -> dict[str, Any]:
    led = _open(ledger)
    try:
        VariableRegistry(led, domain).merge(name, into=into)
    except KeyError as e:
        raise ServiceError(str(e)) from e
    return {"merged": name, "into": VariableRegistry(led, domain).resolve(into)}


# ── Proposals and experiments ─────────────────────────────────────────────────

def _proposal(d: dict[str, Any]) -> Proposal:
    try:
        nv = d.get("new_variable")
        return Proposal(
            variable=d["variable"], description=d.get("description", ""),
            variant_a=_variant(d["variant_a"]), variant_b=_variant(d["variant_b"]),
            tier=int(d.get("tier", 2)), mode=d.get("mode", "interleaved"),
            rationale=d.get("rationale", ""),
            expected_effect=(float(d["expected_effect"])
                             if d.get("expected_effect") is not None else None),
            proposed_by=d.get("proposed_by"),
            new_variable=VariableDef(d["variable"], nv.get("description", ""), (),
                                     nv.get("execution")) if nv else None)
    except (KeyError, TypeError) as e:
        raise ServiceError(f"malformed proposal {d!r}: missing {e}; need variable, "
                           "variant_a, variant_b (label strings or {label, hint})") from e


def _variant(v: Any) -> Variant:
    if isinstance(v, str):
        return Variant(v)
    return Variant(v["label"], v.get("params", {}), v.get("hint"))


class _Fixed:
    name = "manual"

    def __init__(self, proposals: list[Proposal]) -> None:
        self.proposals = proposals

    def propose(self, evidence, n):  # noqa: ANN001 - Proposer protocol
        return self.proposals[:n]


def _reviewed_dict(r: ReviewedProposal) -> dict[str, Any]:
    return {"status": r.status, "variable": r.variable, "flags": list(r.flags),
            "reason": r.reason, "screening": r.screening, "proposal": {
                "variable": r.proposal.variable, "description": r.proposal.description,
                "variant_a": r.proposal.variant_a.to_dict(),
                "variant_b": r.proposal.variant_b.to_dict(),
                "tier": r.proposal.tier, "mode": r.proposal.mode,
                "expected_effect": r.proposal.expected_effect,
                "proposed_by": r.proposal.proposed_by}}


def _screening(cfg: DomainConfig) -> Screening:
    return Screening(max_windows=cfg.max_windows, window=timedelta(days=cfg.window_days),
                     binary=cfg.rule == "proportion")


def review_proposals(ledger: PathLike, domain: str, proposals: list[dict[str, Any]]
                     ) -> list[dict[str, Any]]:
    """Check proposals against the registry and running experiments. Writes nothing."""
    led = _open(ledger)
    cfg = _config(led, domain)
    parsed = [_proposal(p) for p in proposals]
    engine = HypothesisEngine(led, _Fixed(parsed), screening=_screening(cfg))
    return [_reviewed_dict(r) for r in engine.generate(domain, cfg.metric, n=len(parsed))]


def accept_proposal(ledger: PathLike, domain: str, proposal: dict[str, Any], *,
                    start: bool = True, started_at: Optional[str] = None
                    ) -> dict[str, Any]:
    """Review one proposal and, unless rejected, register any new variable, add the
    experiment and (by default) start it now."""
    led = _open(ledger)
    cfg = _config(led, domain)
    engine = HypothesisEngine(led, _Fixed([_proposal(proposal)]), screening=_screening(cfg))
    [reviewed] = engine.generate(domain, cfg.metric, n=1)
    if not reviewed.accepted:
        raise ServiceError(f"proposal rejected: {reviewed.reason}")
    exp = engine.accept(reviewed)
    if start:
        led.start_experiment(exp.id, at=started_at)
    return {"experiment": _experiment_dict(led, led.experiment(exp.id)),
            "review": _reviewed_dict(reviewed)}


def start_experiment(ledger: PathLike, experiment_id: str,
                     started_at: Optional[str] = None) -> dict[str, Any]:
    led = _open(ledger)
    exp = led.experiment(experiment_id)
    if exp is None:
        raise ServiceError(f"unknown experiment {experiment_id!r}")
    if exp.started:
        raise ServiceError(f"experiment {experiment_id} already started at {exp.started}")
    if led.abandoned(experiment_id) is not None:
        raise ServiceError(f"experiment {experiment_id} was abandoned")
    led.start_experiment(experiment_id, at=started_at)
    return _experiment_dict(led, led.experiment(experiment_id))


def _experiment_dict(led: Ledger, exp: Experiment) -> dict[str, Any]:
    final = led.final_decision(exp.id)
    return {"id": exp.id, "domain": exp.domain, "variable": exp.variable,
            "description": exp.description, "variant_a": exp.variant_a.label,
            "variant_b": exp.variant_b.label, "mode": exp.mode, "started": exp.started,
            "status": ("abandoned" if led.abandoned(exp.id) is not None
                       else "not_started" if not exp.started
                       else "closed" if final is not None else "running"),
            "restart_of": exp.restart_of,
            "outcome": final.outcome.value if final else None,
            "observations": len(led.observations(exp.id))}


# ── Data ──────────────────────────────────────────────────────────────────────

def record_observations(ledger: PathLike, observations: list[dict[str, Any]]
                        ) -> dict[str, Any]:
    """Record one row per unit. All rows are validated first; if any is invalid,
    nothing is written. Missing metrics must be null, never 0."""
    led = _open(ledger)
    now = _now().isoformat()
    parsed: list[Observation] = []
    errors: list[str] = []
    for i, row in enumerate(observations):
        try:
            exp = led.experiment(row["experiment_id"])
            if exp is None:
                raise ServiceError(f"unknown experiment {row['experiment_id']!r}")
            if not led.is_open(exp.id):
                raise ServiceError(f"experiment {exp.id} is closed or abandoned")
            if row["variant"] not in (exp.variant_a.label, exp.variant_b.label):
                raise ServiceError(f"variant {row['variant']!r} is not "
                                   f"{exp.variant_a.label!r} or {exp.variant_b.label!r}")
            assigned = led.assignment(exp.id, str(row["unit_id"]))
            if assigned is not None and assigned != row["variant"]:
                raise ServiceError(f"unit {row['unit_id']!r} was assigned {assigned!r}, "
                                   f"not {row['variant']!r}")
            metrics = row.get("metrics") or {}
            for k, v in metrics.items():
                if v is not None and not isinstance(v, (int, float)):
                    raise ServiceError(f"metric {k!r} must be a number or null")
            datetime.fromisoformat(row["produced_at"])
            parsed.append(Observation(
                experiment_id=exp.id, variant=row["variant"], unit_id=str(row["unit_id"]),
                produced_at=row["produced_at"], observed_at=row.get("observed_at") or now,
                metrics={k: (None if v is None else float(v)) for k, v in metrics.items()},
                pair_id=row.get("pair_id"), stratum=row.get("stratum")))
        except (KeyError, ServiceError, ValueError, TypeError) as e:
            msg = f"missing field {e}" if isinstance(e, KeyError) else str(e)
            errors.append(f"row {i}: {msg}")
    if errors:
        raise ServiceError("nothing recorded; fix these rows: " + "; ".join(errors[:20]))
    led.record_observations(parsed)
    return {"recorded": len(parsed)}


# ── Judging ───────────────────────────────────────────────────────────────────

_NEXT = {
    Outcome.B_BETTER: "adopt variant B ({b}); this experiment is closed",
    Outcome.A_BETTER: "keep variant A ({a}); this experiment is closed",
    Outcome.EQUIVALENT: "the difference is below min_effect; keep whichever variant is "
                        "cheaper. This experiment is closed",
    Outcome.NO_DETECTABLE_DIFF: "no clear difference within the time allowed; keep "
                                "variant A ({a}) and move on. This experiment is closed",
    Outcome.INSUFFICIENT: "keep producing units for both variants; do not act on the "
                          "current numbers",
}


def _decision_dict(d: Decision, exp: Experiment, cfg: DomainConfig) -> dict[str, Any]:
    out = {k: v for k, v in asdict(d).items() if k != "rule_params"}
    # the parts of rule_params an agent needs to read the verdict correctly
    out["stratified"] = d.rule_params.get("stratified")
    out["dropped_units"] = d.rule_params.get("dropped_units", 0)
    out["outcome"] = d.outcome.value
    out["interval"] = list(d.interval) if d.interval else None
    out["closed"] = d.outcome.is_final
    out["next_action"] = _NEXT[d.outcome].format(a=exp.variant_a.label, b=exp.variant_b.label)
    if not d.outcome.is_final and exp.started:
        started = datetime.fromisoformat(exp.started)
        started = started if started.tzinfo else started.replace(tzinfo=timezone.utc)
        nxt = started + timedelta(days=cfg.window_days) * (d.checkpoint + 1)
        out["next_checkpoint"] = nxt.isoformat()
        if d.needed_n:
            out["next_action"] += f"; about {d.needed_n} more units per variant likely needed"
    out["settings"] = cfg.to_dict()
    return out


def evaluate(ledger: PathLike, experiment_id: Optional[str] = None, *,
             domain: Optional[str] = None, now: Optional[str] = None) -> list[dict[str, Any]]:
    """Judge one experiment, or every running experiment in *domain*. Safe to call as
    often as you like: verdicts are only taken (and recorded once) at checkpoints."""
    led = _open(ledger)
    at = datetime.fromisoformat(now) if now else _now()
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    if experiment_id:
        exp = led.experiment(experiment_id)
        if exp is None:
            raise ServiceError(f"unknown experiment {experiment_id!r}")
        targets = [exp]
    elif domain:
        targets = [e for e in led.experiments(domain) if e.started
                   and led.is_open(e.id)]
    else:
        raise ServiceError("give an experiment_id or a domain")
    results = []
    for exp in targets:
        cfg = config_for(led, exp)
        if cfg is None:
            raise ServiceError(f"domain {exp.domain!r} is not configured; call configure")
        d = cfg.build_evaluator(led).evaluate(exp.id, cfg.metric, now=at)
        decision = _decision_dict(d, exp, cfg)
        prior = estimate_prior(fd for fd in (led.final_decision(e.id)
                                             for e in led.experiments(exp.domain))
                               if fd is not None)
        # the measured effect, pulled toward 0 for the winner's curse; verdict unchanged
        decision["corrected_effect"] = prior.shrink(d.effect, approx_se(d))
        results.append({"experiment": _experiment_dict(led, exp), "decision": decision})
    return results


def evidence(ledger: PathLike, domain: str, fmt: str = "json") -> Any:
    """What the ledger currently supports, per variable. fmt="markdown" for prompts."""
    led = _open(ledger)
    ev = build_evidence(led, domain, _config(led, domain).metric)
    if fmt == "markdown":
        return ev.to_markdown()
    return {
        "domain": ev.domain, "metric": ev.metric.to_dict(),
        "variables": [{"variable": v.variable, "status": v.status,
                       "best_variant": v.best_variant, "effect": v.effect,
                       "interval": list(v.interval) if v.interval else None,
                       "n_observations": v.n_observations, "needed_n": v.needed_n,
                       "corrected_effect": v.shrunk_effect,
                       "experiments": list(v.experiments),
                       "description": v.definition.description if v.definition else None,
                       "execution": v.definition.execution if v.definition else None}
                      for v in ev.variables],
        "open_experiments": list(ev.open_experiments),
        "metric_dispersion": ev.metric_dispersion, "data_quality": ev.data_quality,
        "cost": ev.cost, "proposers": ev.proposers, "winners_curse": ev.shrinkage,
    }


def status(ledger: PathLike, domain: Optional[str] = None) -> dict[str, Any]:
    led = _open(ledger)
    domains = sorted({e.domain for e in led.experiments()}
                     | {e["config"]["domain"] for e in led.of_kind("domain_config")})
    if domain:
        domains = [domain]
    return {d: {"configured": current_config(led, d) is not None,
                "experiments": [_experiment_dict(led, e) for e in led.experiments(d)]}
            for d in domains}


def calibrate(ledger: PathLike, domain: str, *, effect: float, per_window: int,
              pool: Optional[list[float]] = None, pool_b: Optional[list[float]] = None,
              sims: int = 1000, seed: int = 0) -> dict[str, Any]:
    """How would this domain's settings behave on its own data? effect=0 gives the
    false-winner rate; effect>0 gives how often (and how fast) a real effect is caught.

    pool defaults to every valid value of the domain's metric already in the ledger.
    For 0/1 metrics, pool_b defaults to the same pool with its rate raised by effect."""
    led = _open(ledger)
    cfg = _config(led, domain)
    if pool is None:
        pool = []
        lo, hi = cfg.metric.valid_range
        for exp in led.experiments(domain):
            for obs in led.observations(exp.id):
                v = obs.metrics.get(cfg.metric.name)
                if v is None or not math.isfinite(v):
                    continue
                if (lo is not None and v < lo) or (hi is not None and v > hi):
                    continue
                pool.append(v)
    if len(pool) < 10:
        raise ServiceError(f"only {len(pool)} usable values; pass a pool of historical "
                           "values (at least 10) or record more data first")
    if cfg.rule == "proportion" and pool_b is None and effect != 0:
        rate = statistics.fmean(pool) + (effect if cfg.metric.higher_is_better else -effect)
        if not 0.0 <= rate <= 1.0:
            raise ServiceError("base rate plus effect falls outside 0–1")
        ones = round(rate * 1000)
        pool_b = [1.0] * ones + [0.0] * (1000 - ones)
    c = _calibrate(pool, cfg.build_rule(), cfg.metric, effect=effect, per_window=per_window,
                   windows=cfg.max_windows, sims=sims, seed=seed, pool_b=pool_b)
    return {"effect": effect, "per_window": per_window, "sims": sims,
            "pool_size": len(pool), "false_winner_rate" if effect == 0 else "detection_rate":
            c.wrong_direction if effect == 0 else c.detected,
            "wrong_direction_rate": c.wrong_direction,
            "median_windows_to_verdict": c.median_weeks_to_final,
            "outcomes": c.outcomes, "settings": cfg.to_dict()}


# ── Assignment and putting verdicts into effect ────────────────────────────────

def assign_variant(ledger: PathLike, experiment_id: str, unit_id: str,
                   stratum: Optional[str] = None) -> dict[str, Any]:
    """Which variant a unit should get. Balanced within its stratum, recorded, and
    stable: asking again for the same unit gives the same answer."""
    led = _open(ledger)
    try:
        label = _assign(led, experiment_id, unit_id, stratum)
    except (KeyError, ValueError) as e:
        raise ServiceError(str(e).strip("'\"")) from e
    exp = led.experiment(experiment_id)
    variant = exp.variant_a if label == exp.variant_a.label else exp.variant_b
    return {"experiment_id": experiment_id, "unit_id": unit_id, "stratum": stratum,
            **variant.to_dict()}


def pending(ledger: PathLike, domain: str) -> list[dict[str, Any]]:
    """Closed experiments whose verdict hasn't been put into effect yet."""
    led = _open(ledger)
    out = []
    for exp in led.experiments(domain):
        d = led.final_decision(exp.id)
        if d is None or led.applied(exp.id) is not None:
            continue
        b_won = d.outcome is Outcome.B_BETTER
        out.append({"experiment_id": exp.id, "variable": exp.variable,
                    "outcome": d.outcome.value,
                    "variant": exp.variant_b.label if b_won else exp.variant_a.label,
                    "changes_pipeline": b_won, "effect": d.effect,
                    "interval": list(d.interval) if d.interval else None})
    return out


def mark_applied(ledger: PathLike, experiment_id: str) -> dict[str, Any]:
    """Record that a closed experiment's verdict is now in effect in the pipeline."""
    led = _open(ledger)
    exp = led.experiment(experiment_id)
    d = led.final_decision(experiment_id) if exp else None
    if exp is None or d is None:
        raise ServiceError(f"{experiment_id!r} is not a closed experiment")
    variant = exp.variant_b.label if d.outcome is Outcome.B_BETTER else exp.variant_a.label
    try:
        led.mark_applied(experiment_id, variant, d.outcome.value)
    except ValueError as e:
        raise ServiceError(str(e)) from e
    return {"experiment_id": experiment_id, "variant": variant, "outcome": d.outcome.value}


# ── Lifecycle ──────────────────────────────────────────────────────────────────

def abandon_experiment(ledger: PathLike, experiment_id: str, reason: str) -> dict[str, Any]:
    """Close an experiment with no verdict (its data is no longer comparable, or it was
    set up wrong). It is never judged or applied and frees its variable and slot."""
    led = _open(ledger)
    try:
        _abandon(led, experiment_id, reason)
    except (KeyError, ValueError) as e:
        raise ServiceError(str(e).strip("'\"")) from e
    return _experiment_dict(led, led.experiment(experiment_id))


def restart_experiment(ledger: PathLike, experiment_id: str, reason: str, *,
                       at: Optional[str] = None, start: bool = True) -> dict[str, Any]:
    """Abandon an experiment and register it again from *at* (default now) — e.g. after
    the pipeline changed mid-experiment. Only data from after *at* will count."""
    led = _open(ledger)
    try:
        new = _restart(led, experiment_id, reason, at=at, start=start)
    except (KeyError, ValueError) as e:
        raise ServiceError(str(e).strip("'\"")) from e
    return {"abandoned": _experiment_dict(led, led.experiment(experiment_id)),
            "experiment": _experiment_dict(led, new)}
