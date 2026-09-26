"""ledger.py — The append-only ledger, the single source of truth.

Every event has an envelope {"schema": 2, "kind": ..., "recorded_at": ...}:

    experiment          an Experiment definition (Experiment.to_dict())
    experiment_started  {"experiment_id", "at"}
    observation         Observation.to_dict()
    decision            Decision.to_dict()
    variable            {"domain", "name", "description", "aliases", "execution"}
    variable_alias      {"domain", "alias", "name"}
    legacy_arm_summary  a v0.1 record (arm average + caller-supplied winner flag)
    assignment          {"experiment_id", "unit_id", "variant", "stratum"} (assignment.py)
    applied             {"experiment_id", "variant", "outcome"}: a verdict put into effect
    abandoned           {"experiment_id", "reason"}: closed without a verdict (lifecycle.py)

Nothing is ever rewritten; state is derived by replaying events. Where events are
kept is an EventLog (eventlog.py); when "now" is comes from a Clock (clock.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Union

from .clock import Clock, system_clock
from .decision import Decision
from .eventlog import SCHEMA, EventLog, JsonlFile, LedgerReadOnlyError, MemoryLog
from .experiment import Experiment
from .metrics import Observation

__all__ = ["Ledger", "LedgerReadOnlyError", "SCHEMA"]


def _as_log(log: Union[EventLog, Path, str, None]) -> EventLog:
    """A path means a JSONL file there; None means memory only."""
    if log is None:
        return MemoryLog()
    if isinstance(log, (str, Path)):
        return JsonlFile(log)
    return log


class Ledger:
    def __init__(self, log: Union[EventLog, Path, str, None] = None, *,
                 clock: Clock = system_clock) -> None:
        self.log = _as_log(log)
        self.clock = clock
        self._events: list[dict[str, Any]] = self.log.load()

    @property
    def read_only(self) -> bool:
        return self.log.read_only

    def now(self) -> str:
        return self.clock().isoformat()

    # ── Write ──────────────────────────────────────────────────────────────────

    def _append(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        event = {"schema": SCHEMA, "kind": kind, "recorded_at": self.now(), **payload}
        self.log.append(event)            # raises LedgerReadOnlyError for a v0.1 file
        self._events.append(event)
        return event

    def add_experiment(self, experiment: Experiment) -> None:
        if self.experiment(experiment.id) is not None:
            raise ValueError(f"experiment {experiment.id} already in ledger")
        self._append("experiment", {"experiment": experiment.to_dict()})

    def start_experiment(self, experiment_id: str, at: Optional[str] = None) -> None:
        if self.experiment(experiment_id) is None:
            raise KeyError(experiment_id)
        self._append("experiment_started", {"experiment_id": experiment_id,
                                             "at": at or self.now()})

    def record_observation(self, obs: Observation) -> None:
        if self.experiment(obs.experiment_id) is None:
            raise KeyError(f"unknown experiment {obs.experiment_id}")
        if self.abandoned(obs.experiment_id) is not None:
            raise ValueError(f"experiment {obs.experiment_id} was abandoned")
        assigned = self.assignment(obs.experiment_id, obs.unit_id)
        if assigned is not None and assigned != obs.variant:
            raise ValueError(f"unit {obs.unit_id!r} was assigned {assigned!r}, not "
                             f"{obs.variant!r}; record what the unit actually received")
        self._append("observation", obs.to_dict())

    def record_observations(self, observations: Iterable[Observation]) -> None:
        for obs in observations:
            self.record_observation(obs)

    def record_decision(self, decision: Decision) -> None:
        self._append("decision", decision.to_dict())

    def append_raw(self, kind: str, payload: dict[str, Any]) -> None:
        """Low-level append, for components that own their own event kinds (registry)."""
        self._append(kind, payload)

    # ── Read ───────────────────────────────────────────────────────────────────

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self._events if e.get("kind") == kind]

    def experiments(self, domain: Optional[str] = None) -> list[Experiment]:
        started = {e["experiment_id"]: e["at"] for e in self.of_kind("experiment_started")}
        result = []
        for e in self.of_kind("experiment"):
            exp = Experiment.from_dict(e["experiment"])
            if domain is not None and exp.domain != domain:
                continue
            if exp.id in started:
                exp.started = started[exp.id]
            result.append(exp)
        return result

    def experiment(self, experiment_id: str) -> Optional[Experiment]:
        for exp in self.experiments():
            if exp.id == experiment_id:
                return exp
        return None

    def observations(self, experiment_id: str) -> list[Observation]:
        """Latest observation per unit for this experiment."""
        latest: dict[str, Observation] = {}
        for e in self.of_kind("observation"):
            if e["experiment_id"] != experiment_id:
                continue
            obs = Observation.from_dict(e)
            prev = latest.get(obs.unit_id)
            if prev is None or obs.observed_at >= prev.observed_at:
                latest[obs.unit_id] = obs
        return list(latest.values())

    def decisions(self, experiment_id: Optional[str] = None) -> list[Decision]:
        return [Decision.from_dict(e) for e in self.of_kind("decision")
                if experiment_id is None or e["experiment_id"] == experiment_id]

    def final_decision(self, experiment_id: str) -> Optional[Decision]:
        finals = [d for d in self.decisions(experiment_id) if d.outcome.is_final]
        return finals[-1] if finals else None

    def assignment(self, experiment_id: str, unit_id: str) -> Optional[str]:
        for e in reversed(self._events):
            if (e.get("kind") == "assignment" and e["experiment_id"] == experiment_id
                    and e["unit_id"] == unit_id):
                return e["variant"]
        return None

    def mark_applied(self, experiment_id: str, variant: str, outcome: str) -> None:
        if self.applied(experiment_id) is not None:
            raise ValueError(f"experiment {experiment_id} was already applied")
        self._append("applied", {"experiment_id": experiment_id, "variant": variant,
                                 "outcome": outcome})

    def abandoned(self, experiment_id: str) -> Optional[dict[str, Any]]:
        for e in self.of_kind("abandoned"):
            if e["experiment_id"] == experiment_id:
                return e
        return None

    def is_open(self, experiment_id: str) -> bool:
        """Neither judged to a final verdict nor abandoned."""
        return (self.final_decision(experiment_id) is None
                and self.abandoned(experiment_id) is None)

    def applied(self, experiment_id: str) -> Optional[dict[str, Any]]:
        for e in self.of_kind("applied"):
            if e["experiment_id"] == experiment_id:
                return e
        return None

    def legacy_records(self, domain: Optional[str] = None) -> list[dict[str, Any]]:
        return [e for e in self.of_kind("legacy_arm_summary")
                if domain is None or e.get("domain") == domain]

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        mode = ", read_only" if self.read_only else ""
        return f"Ledger({self.log.describe()}, events={len(self._events)}{mode})"
