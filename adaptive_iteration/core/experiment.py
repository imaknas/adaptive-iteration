"""core/experiment.py — Experiment / Variant dataclasses + ExperimentState.

All fields are plain Python types so they round-trip cleanly through JSON.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class Variant:
    """One arm of an experiment."""
    label: str
    params: dict[str, Any] = field(default_factory=dict)
    hint: Optional[str] = None          # short text cue for downstream renderers

    def to_dict(self) -> dict:
        return {"label": self.label, "params": self.params, "hint": self.hint}

    @classmethod
    def from_dict(cls, d: dict) -> "Variant":
        return cls(label=d["label"], params=d.get("params", {}), hint=d.get("hint"))


@dataclass
class Experiment:
    """A single testable hypothesis with two variants (A/B).

    Fields
    ------
    id          : short unique id (8-char hex by default)
    domain      : e.g. "short_video", "blog_post", "email_campaign"
    variable    : what's being tested, e.g. "hook_type", "cta_style"
    description : human-readable summary of the hypothesis
    variant_a   : control arm
    variant_b   : challenger arm
    tier        : priority tier (1 = run alone, 2 = can run in parallel, 3 = no A/B)
    started     : ISO date string when experiment went active, or None
    concluded   : ISO date string when experiment was evaluated, or None
    status      : "pending" | "active" | "concluded"
    items       : list of {"item_id": str, "variant": str, ...} — produced units
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    domain: str = ""
    variable: str = ""
    description: str = ""
    variant_a: Variant = field(default_factory=lambda: Variant(label="control"))
    variant_b: Variant = field(default_factory=lambda: Variant(label="variant"))
    tier: int = 1
    started: Optional[str] = None
    concluded: Optional[str] = None
    status: str = "pending"
    mode: Literal["interleaved", "paired"] = "interleaved"
    """Experiment execution mode.

    - ``interleaved``: Different topics are assigned to variants in alternating
      fashion.  Maximises throughput (Tier 2 experiments).  One topic → one
      video → one variant.

    - ``paired``: The *same* topic is used to produce two videos — one per
      variant — in a single run.  This eliminates topic-level confounders and
      yields cleaner causal inference.  Preferred for Tier 1 experiments where
      the variable under test (e.g. subscribe_prompt, cta_style) is independent
      of the topic itself.
    """
    items: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id":          self.id,
            "domain":      self.domain,
            "variable":    self.variable,
            "description": self.description,
            "variant_a":   self.variant_a.to_dict(),
            "variant_b":   self.variant_b.to_dict(),
            "tier":        self.tier,
            "started":     self.started,
            "concluded":   self.concluded,
            "status":      self.status,
            "mode":        self.mode,
            "items":       self.items,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Experiment":
        return cls(
            id=d.get("id", uuid.uuid4().hex[:8]),
            domain=d.get("domain", ""),
            variable=d.get("variable", ""),
            description=d.get("description", ""),
            variant_a=Variant.from_dict(d.get("variant_a", {"label": "control"})),
            variant_b=Variant.from_dict(d.get("variant_b", {"label": "variant"})),
            tier=d.get("tier", 1),
            started=d.get("started"),
            concluded=d.get("concluded"),
            status=d.get("status", "pending"),
            mode=d.get("mode", "interleaved"),
            items=d.get("items", []),
        )


@dataclass
class ExperimentState:
    """Persisted state for one domain's experiment pipeline."""
    current: Optional[Experiment] = None
    queue: list[Experiment] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "current": self.current.to_dict() if self.current else {},
            "queue":   [e.to_dict() for e in self.queue],
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentState":
        current_raw = d.get("current", {})
        return cls(
            current=Experiment.from_dict(current_raw) if current_raw else None,
            queue=[Experiment.from_dict(e) for e in d.get("queue", [])],
            history=d.get("history", []),
        )
