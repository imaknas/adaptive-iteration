"""migrate.py — Convert a v0.1 ledger (JSON array) into a v0.2 JSONL ledger.

v0.1 records only hold per-arm averages plus a caller-supplied winner flag, so they
cannot be re-judged. They are carried over as legacy_arm_summary events and every
variable name that appears is registered, so later experiments build on the same
vocabulary. The source file is never modified.
"""
from __future__ import annotations

import json
from pathlib import Path

from .core.ledger import Ledger, legacy_event
from .core.registry import VariableDef, VariableRegistry


def v1_to_v2(src: Path, dst: Path) -> Ledger:
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; refusing to overwrite")
    records = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{src} is not a v0.1 ledger (expected a JSON array)")

    ledger = Ledger(dst)
    for r in records:
        event = legacy_event(r)
        ledger.append_raw("legacy_arm_summary",
                          {k: v for k, v in event.items()
                           if k not in ("schema", "kind", "recorded_at")}
                          | {"original_timestamp": r.get("timestamp")})
    seen: set[tuple[str, str]] = set()
    for r in records:
        domain, variable = r.get("domain"), r.get("variable")
        if not domain or not variable or (domain, variable) in seen:
            continue
        seen.add((domain, variable))
        registry = VariableRegistry(ledger, domain)
        if registry.resolve(variable) is None:
            registry.register(VariableDef(name=variable,
                                          description="(migrated from v0.1 ledger)"))
    return ledger
