"""core/ledger.py — Append-only JSONL ledger, the single source of truth.

Every line is one event with an envelope {"schema": 2, "kind": ..., "recorded_at": ...}:

    experiment          an Experiment definition (Experiment.to_dict())
    experiment_started  {"experiment_id", "at"}
    observation         Observation.to_dict()
    decision            Decision.to_dict()
    variable            {"domain", "name", "description", "aliases", "execution"}
    variable_alias      {"domain", "alias", "name"}
    legacy_arm_summary  a v0.1 record (arm average + caller-supplied winner flag)

Nothing is ever rewritten; state is derived by replaying events. A v0.1 ledger
(a single JSON array) opens read-only — convert it with adaptive_iteration.migrate.
Ledger(None) keeps events in memory only (used by replay).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .decision import Decision
from .experiment import Experiment
from .metrics import Observation

SCHEMA = 2


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LedgerReadOnlyError(RuntimeError):
    pass


class Ledger:
    def __init__(self, path: Optional[Path]) -> None:
        self.path = Path(path) if path is not None else None
        self._events: list[dict[str, Any]] = []
        self.read_only = False
        if self.path is not None and self.path.exists():
            text = self.path.read_text(encoding="utf-8")
            if text.lstrip().startswith("["):
                self.read_only = True
                self._events = [legacy_event(r) for r in json.loads(text)]
            else:
                self._events = [json.loads(line) for line in text.splitlines() if line.strip()]

    # ── Write ──────────────────────────────────────────────────────────────────

    def _append(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.read_only:
            raise LedgerReadOnlyError(
                f"{self.path} is a v0.1 ledger (JSON array); convert it with "
                "adaptive_iteration.migrate.v1_to_v2() before writing"
            )
        event = {"schema": SCHEMA, "kind": kind, "recorded_at": utcnow_iso(), **payload}
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
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
                                             "at": at or utcnow_iso()})

    def record_observation(self, obs: Observation) -> None:
        if self.experiment(obs.experiment_id) is None:
            raise KeyError(f"unknown experiment {obs.experiment_id}")
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

    def legacy_records(self, domain: Optional[str] = None) -> list[dict[str, Any]]:
        return [e for e in self.of_kind("legacy_arm_summary")
                if domain is None or e.get("domain") == domain]

    def __len__(self) -> int:
        return len(self._events)

    def __repr__(self) -> str:
        mode = ", read_only" if self.read_only else ""
        return f"Ledger(path={self.path!r}, events={len(self._events)}{mode})"


def legacy_event(record: dict[str, Any]) -> dict[str, Any]:
    """Wrap a v0.1 ledger record as a v2 legacy_arm_summary event."""
    return {
        "schema": SCHEMA,
        "kind": "legacy_arm_summary",
        "recorded_at": record.get("timestamp"),
        "domain": record.get("domain"),
        "experiment_id": record.get("experiment_id"),
        "variable": record.get("variable"),
        "variant": record.get("variant"),
        "metric_values": record.get("metric_values", {}),
        "winner": record.get("winner"),
    }
