"""core/metrics.py — MetricSpec (what counts as better) and Observation (one unit's data)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class MetricSpec:
    """Definition of the metric an experiment is judged on.

    min_effect is the smallest difference that matters in this domain (the half-width
    of the region of practical equivalence). It has no default on purpose: without it
    "no difference" and "don't know yet" cannot be told apart.
    """
    name: str
    min_effect: float
    higher_is_better: bool = True
    valid_range: tuple[Optional[float], Optional[float]] = (None, None)

    def __post_init__(self) -> None:
        if self.min_effect <= 0:
            raise ValueError("min_effect must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_effect": self.min_effect,
            "higher_is_better": self.higher_is_better,
            "valid_range": list(self.valid_range),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetricSpec":
        return cls(
            name=d["name"],
            min_effect=d["min_effect"],
            higher_is_better=d.get("higher_is_better", True),
            valid_range=tuple(d.get("valid_range", (None, None))),
        )


@dataclass(frozen=True)
class Observation:
    """One measurement of one produced unit (a video, an email, a proposal…).

    produced_at : when the unit went out — maturity is measured from here
    observed_at : when these metric values were read
    metrics     : None means "no data", never zero. Adapters must not fill gaps with 0.
    pair_id     : paired experiments — the two units made from the same input
    stratum     : optional group the unit belongs to (e.g. topic category); rules that
                  support it compare arms within each stratum to remove mix effects

    A unit may be observed several times as its metrics mature; the latest
    observation per (experiment_id, unit_id) wins.
    """
    experiment_id: str
    variant: str
    unit_id: str
    produced_at: str
    observed_at: str
    metrics: dict[str, Optional[float]] = field(default_factory=dict)
    pair_id: Optional[str] = None
    stratum: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "variant": self.variant,
            "unit_id": self.unit_id,
            "produced_at": self.produced_at,
            "observed_at": self.observed_at,
            "metrics": dict(self.metrics),
            "pair_id": self.pair_id,
            "stratum": self.stratum,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Observation":
        return cls(
            experiment_id=d["experiment_id"],
            variant=d["variant"],
            unit_id=d["unit_id"],
            produced_at=d["produced_at"],
            observed_at=d["observed_at"],
            metrics=dict(d.get("metrics", {})),
            pair_id=d.get("pair_id"),
            stratum=d.get("stratum"),
        )
