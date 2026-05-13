"""core/config.py — AdaptiveConfig: read/write JSON config with winner hints support.

Config file schema (example):
{
    "domain": "short_video",
    "primary_metric": "avg_view_pct",
    "min_items_per_variant": 3,
    "winner_hints": {
        "hook_type": "question",
        "cta_style": "soft"
    },
    "extra": {}
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


_DEFAULTS: dict[str, Any] = {
    "domain":                "default",
    "primary_metric":        "primary_metric",
    "min_items_per_variant": 3,
    "winner_hints":          {},
    "extra":                 {},
}


class AdaptiveConfig:
    """Lightweight JSON config bag with typed accessor helpers.

    Parameters
    ----------
    path : where to load from / save to (created on first save if absent)
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, Any] = dict(_DEFAULTS)
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                self._data.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._flush()

    def update(self, mapping: dict[str, Any]) -> None:
        self._data.update(mapping)
        self._flush()

    # ── Winner hints ───────────────────────────────────────────────────────────

    def set_winner_hint(self, variable: str, variant_label: str) -> None:
        """Record which variant won for *variable*."""
        hints = self._data.setdefault("winner_hints", {})
        hints[variable] = variant_label
        self._flush()

    def get_winner_hint(self, variable: str) -> Optional[str]:
        return self._data.get("winner_hints", {}).get(variable)

    def all_winner_hints(self) -> dict[str, str]:
        return dict(self._data.get("winner_hints", {}))

    # ── Convenience ────────────────────────────────────────────────────────────

    @property
    def domain(self) -> str:
        return str(self._data.get("domain", "default"))

    @property
    def primary_metric(self) -> str:
        return str(self._data.get("primary_metric", "primary_metric"))

    @property
    def min_items_per_variant(self) -> int:
        return int(self._data.get("min_items_per_variant", 3))

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    # ── Persistence ────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def save(self) -> None:
        """Explicit save (normally auto-saved on every set/update)."""
        self._flush()

    def __repr__(self) -> str:
        return f"AdaptiveConfig(path={self.path!r}, domain={self.domain!r})"
