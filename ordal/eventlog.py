"""eventlog.py — Where ledger events are kept.

The Ledger only needs to read every event back and append new ones; an EventLog
provides exactly that. JsonlFile keeps one event per line in a file; MemoryLog keeps
them in memory (replay, tests). A database-backed log would be a third.

A v0.1 ledger file (a single JSON array) loads as legacy events and is read-only;
ordal.migrate converts it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

SCHEMA = 2


class LedgerReadOnlyError(RuntimeError):
    pass


class EventLog(Protocol):
    read_only: bool

    def load(self) -> list[dict[str, Any]]:
        """Every stored event, oldest first."""
        ...

    def append(self, event: dict[str, Any]) -> None: ...

    def describe(self) -> str:
        """Human-readable location, for messages."""
        ...


class JsonlFile:
    """One JSON event per line; created on first write."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.read_only = self.path.exists() and self._is_v1()

    def _is_v1(self) -> bool:
        with self.path.open(encoding="utf-8") as f:
            head = f.read(64)
        return head.lstrip().startswith("[")

    def load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        if self.read_only:
            return [legacy_event(r) for r in json.loads(text)]
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def append(self, event: dict[str, Any]) -> None:
        if self.read_only:
            raise LedgerReadOnlyError(
                f"{self.path} is a v0.1 ledger (JSON array); convert it with "
                "ordal.migrate.v1_to_v2() before writing")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def describe(self) -> str:
        return str(self.path)


class MemoryLog:
    """Events kept in memory only."""

    read_only = False

    def __init__(self, events: Optional[Iterable[dict[str, Any]]] = None) -> None:
        self.events: list[dict[str, Any]] = list(events or [])

    def load(self) -> list[dict[str, Any]]:
        return list(self.events)

    def append(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def describe(self) -> str:
        return "memory"


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
