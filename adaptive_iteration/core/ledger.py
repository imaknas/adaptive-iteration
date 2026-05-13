"""core/ledger.py — Domain-agnostic JSON ledger.

Replaces results.tsv with a structured, queryable JSON store.

Record schema
-------------
{
    "domain":        str,          # e.g. "short_video"
    "experiment_id": str,
    "variable":      str,          # what was tested
    "variant":       str,          # variant label ("control" / "challenger" / ...)
    "metric_values": dict,         # raw metric snapshot, e.g. {"avg_view_pct": 45.2}
    "winner":        bool | None,  # True if this variant won
    "timestamp":     str           # ISO-8601 datetime
}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class Ledger:
    """Append-only JSON ledger stored at *path*.

    Usage
    -----
    ledger = Ledger(Path("data/ledger.json"))
    ledger.append(
        domain="short_video",
        experiment_id="abc123",
        variable="hook_type",
        variant="question",
        metric_values={"avg_view_pct": 52.1, "like_rate": 3.2},
        winner=True,
    )
    records = ledger.query(domain="short_video")
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._records: list[dict[str, Any]] = []
        if path.exists():
            try:
                self._records = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._records = []

    # ── Write ──────────────────────────────────────────────────────────────────

    def append(
        self,
        domain: str,
        experiment_id: str,
        variable: str,
        variant: str,
        metric_values: dict[str, Any],
        winner: Optional[bool] = None,
        timestamp: Optional[str] = None,
    ) -> dict[str, Any]:
        record = {
            "domain":        domain,
            "experiment_id": experiment_id,
            "variable":      variable,
            "variant":       variant,
            "metric_values": metric_values,
            "winner":        winner,
            "timestamp":     timestamp or datetime.now(timezone.utc).isoformat(),
        }
        self._records.append(record)
        self._flush()
        return record

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def query(
        self,
        domain: Optional[str] = None,
        variable: Optional[str] = None,
        experiment_id: Optional[str] = None,
        winners_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Return filtered records. All filters are ANDed together."""
        result = self._records
        if domain is not None:
            result = [r for r in result if r.get("domain") == domain]
        if variable is not None:
            result = [r for r in result if r.get("variable") == variable]
        if experiment_id is not None:
            result = [r for r in result if r.get("experiment_id") == experiment_id]
        if winners_only:
            result = [r for r in result if r.get("winner") is True]
        return list(result)

    def all_domains(self) -> list[str]:
        return sorted({r["domain"] for r in self._records if "domain" in r})

    def all_variables(self, domain: Optional[str] = None) -> list[str]:
        records = self.query(domain=domain) if domain else self._records
        return sorted({r["variable"] for r in records if "variable" in r})

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"Ledger(path={self.path!r}, records={len(self._records)})"
