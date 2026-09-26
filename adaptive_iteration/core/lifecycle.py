"""core/lifecycle.py — Ending an experiment without a verdict, and starting it over.

Sometimes an experiment's data stops being comparable before it can be judged: the
pipeline changed underneath it (new model, new prompt, new config version), or it
was set up wrong. Pretending it reached a verdict would be a lie, and leaving it
open would block its variable and its slot forever. So:

abandon()  closes it with no verdict. It is never judged or applied, takes no more
           data, frees its variable and slot, and shows as "abandoned" in evidence.
restart()  abandons it and registers the same definition again with a new start
           time, so it is judged only on data from after the change — and under the
           domain settings in effect from then on. Assignments don't carry over.
"""
from __future__ import annotations

from typing import Optional

from .experiment import Experiment
from .ledger import Ledger


def abandon(ledger: Ledger, experiment_id: str, reason: str) -> None:
    if ledger.experiment(experiment_id) is None:
        raise KeyError(f"unknown experiment {experiment_id!r}")
    if ledger.final_decision(experiment_id) is not None:
        raise ValueError(f"experiment {experiment_id} already has a verdict")
    if ledger.abandoned(experiment_id) is not None:
        raise ValueError(f"experiment {experiment_id} was already abandoned")
    if not reason.strip():
        raise ValueError("give a reason; it is the only record of why this was dropped")
    ledger.append_raw("abandoned", {"experiment_id": experiment_id, "reason": reason})


def restart(ledger: Ledger, experiment_id: str, reason: str, *,
            at: Optional[str] = None, start: bool = True) -> Experiment:
    """Abandon *experiment_id* and register the same experiment again. start=False
    leaves the new one waiting (e.g. for approval); otherwise it starts at *at*
    (default now)."""
    old = ledger.experiment(experiment_id)
    if old is None:
        raise KeyError(f"unknown experiment {experiment_id!r}")
    abandon(ledger, experiment_id, reason)
    new = Experiment(domain=old.domain, variable=old.variable, description=old.description,
                     variant_a=old.variant_a, variant_b=old.variant_b, tier=old.tier,
                     mode=old.mode, proposed_by=old.proposed_by,
                     expected_effect=old.expected_effect, restart_of=old.id)
    ledger.add_experiment(new)
    if start:
        ledger.start_experiment(new.id, at=at)
    return ledger.experiment(new.id)
