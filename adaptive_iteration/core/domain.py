"""core/domain.py — Per-domain judging settings, stored in the ledger.

Recording the metric, rule and schedule once per domain means callers (especially
agents) don't restate them on every call — and can't quietly change them. Settings
are append-only "domain_config" events, and an experiment is always judged with the
settings that were in effect when it started: relaxing min_effect later only affects
experiments started after the change. No moving the goalposts mid-experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional

from .decision import DecisionRule, ProportionIntervalRule, WelchIntervalRule
from .evaluator import Evaluator
from .experiment import Experiment
from .ledger import Ledger
from .metrics import MetricSpec

RuleName = Literal["welch", "proportion"]


@dataclass(frozen=True)
class DomainConfig:
    domain: str
    metric: MetricSpec
    rule: RuleName = "welch"
    superiority: Literal["significance", "margin"] = "significance"
    window_days: float = 7.0
    maturity_hours: float = 72.0
    max_windows: int = 4

    def __post_init__(self) -> None:
        if self.rule not in ("welch", "proportion"):
            raise ValueError("rule must be 'welch' or 'proportion'")
        if self.window_days <= 0 or self.maturity_hours < 0 or self.max_windows < 1:
            raise ValueError("window_days > 0, maturity_hours >= 0, max_windows >= 1")

    def build_rule(self) -> DecisionRule:
        if self.rule == "proportion":
            return ProportionIntervalRule(superiority=self.superiority)
        return WelchIntervalRule(superiority=self.superiority)

    def build_evaluator(self, ledger: Ledger) -> Evaluator:
        return Evaluator(ledger, rule=self.build_rule(),
                         window=timedelta(days=self.window_days),
                         maturity=timedelta(hours=self.maturity_hours),
                         max_windows=self.max_windows)

    def to_dict(self) -> dict[str, Any]:
        return {"domain": self.domain, "metric": self.metric.to_dict(), "rule": self.rule,
                "superiority": self.superiority, "window_days": self.window_days,
                "maturity_hours": self.maturity_hours, "max_windows": self.max_windows}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DomainConfig":
        return cls(domain=d["domain"], metric=MetricSpec.from_dict(d["metric"]),
                   rule=d.get("rule", "welch"), superiority=d.get("superiority", "significance"),
                   window_days=d.get("window_days", 7.0),
                   maturity_hours=d.get("maturity_hours", 72.0),
                   max_windows=d.get("max_windows", 4))


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def save_config(ledger: Ledger, config: DomainConfig) -> None:
    ledger.append_raw("domain_config", {"config": config.to_dict()})


def config_history(ledger: Ledger, domain: str) -> list[tuple[str, DomainConfig]]:
    return [(e["recorded_at"], DomainConfig.from_dict(e["config"]))
            for e in ledger.of_kind("domain_config") if e["config"]["domain"] == domain]


def current_config(ledger: Ledger, domain: str) -> Optional[DomainConfig]:
    history = config_history(ledger, domain)
    return history[-1][1] if history else None


def config_for(ledger: Ledger, experiment: Experiment) -> Optional[DomainConfig]:
    """Settings in effect when *experiment* started. If the domain was configured only
    afterwards, the first configuration recorded applies (never a later edit)."""
    history = config_history(ledger, experiment.domain)
    if not history:
        return None
    if not experiment.started:
        return history[-1][1]
    started = _parse(experiment.started)
    before = [cfg for at, cfg in history if _parse(at) <= started]
    return before[-1] if before else history[0][1]
