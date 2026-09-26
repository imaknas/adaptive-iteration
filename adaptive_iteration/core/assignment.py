"""core/assignment.py — Which variant should the next unit get?

Letting a pipeline pick variants itself is where bias creeps in ("this topic suits
B"). assign() decides instead: within the unit's stratum it gives the next unit to
whichever variant has fewer so far, and breaks ties with a hash of the unit id — so
the mix of strata stays balanced between arms and every decision is reproducible.

Each assignment is written to the ledger ("assignment" events). Asking again for the
same unit returns the same answer, and the ledger refuses an observation whose
variant contradicts the unit's assignment.
"""
from __future__ import annotations

import hashlib
from typing import Optional

from .ledger import Ledger


def assign(ledger: Ledger, experiment_id: str, unit_id: str,
           stratum: Optional[str] = None) -> str:
    exp = ledger.experiment(experiment_id)
    if exp is None:
        raise KeyError(f"unknown experiment {experiment_id!r}")
    if exp.mode == "paired":
        raise ValueError("paired experiments produce both variants for every pair_id; "
                         "there is nothing to assign")
    if not exp.started:
        raise ValueError(f"experiment {experiment_id} has not started")
    if not ledger.is_open(experiment_id):
        raise ValueError(f"experiment {experiment_id} is closed")

    existing = ledger.assignment(experiment_id, unit_id)
    if existing is not None:
        return existing

    labels = (exp.variant_a.label, exp.variant_b.label)
    counts = {label: 0 for label in labels}
    for e in ledger.of_kind("assignment"):
        if e["experiment_id"] == experiment_id and e.get("stratum") == stratum:
            counts[e["variant"]] += 1
    if counts[labels[0]] != counts[labels[1]]:
        variant = min(labels, key=lambda label: counts[label])
    else:
        digest = hashlib.sha256(f"{experiment_id}:{unit_id}".encode()).digest()
        variant = labels[digest[0] & 1]
    ledger.append_raw("assignment", {"experiment_id": experiment_id, "unit_id": unit_id,
                                     "variant": variant, "stratum": stratum})
    return variant
