"""adapters/base.py — DomainAdapter: the only layer that touches external systems.

An adapter turns an experiment's produced units into Observations. It must report
unavailable metrics as None — never as 0 — so the Evaluator can exclude them.
"""
from __future__ import annotations

from typing import Protocol

from ..core.experiment import Experiment
from ..core.metrics import Observation


class DomainAdapter(Protocol):
    def collect_observations(self, experiment: Experiment) -> list[Observation]:
        """Read current metrics for every unit produced under *experiment*."""
        ...
