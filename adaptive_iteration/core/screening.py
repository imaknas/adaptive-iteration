"""core/screening.py — Would this hypothesis even be detectable here?

Nobody can know in advance whether a hypothesis is right. But given the effect the
proposer expects, the domain's own spread and how many units it produces per window,
one can know whether an experiment could possibly reach a verdict before it runs out
of windows. Testing something that can't be detected wastes weeks either way.

Estimates come from the domain's recorded observations. With too little data the
answer is "unknown", never a guess.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import NormalDist
from typing import Any, Optional

from .ledger import Ledger
from .metrics import MetricSpec

MIN_VALUES = 10


@dataclass(frozen=True)
class Screening:
    """Settings the screen needs; normally taken from the domain's configuration."""
    alpha: float = 0.05
    power: float = 0.8
    max_windows: int = 4
    window: timedelta = timedelta(days=7)
    binary: bool = False
    units_per_arm_per_window: Optional[float] = None   # None: estimate from the ledger


@dataclass(frozen=True)
class Capacity:
    """What the domain's data says about spread and volume (None = not enough data)."""
    sd: Optional[float]
    units_per_arm_per_window: Optional[float]
    basis: str

    def needed_per_arm(self, effect: float, screening: Screening) -> Optional[int]:
        if not self.sd or effect == 0:
            return None
        alpha = screening.alpha / screening.max_windows
        z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(screening.power)
        return math.ceil(2 * (z * self.sd / abs(effect)) ** 2)

    def assess(self, effect: Optional[float], spec: MetricSpec, screening: Screening
               ) -> dict[str, Any]:
        """Verdict on one expected effect: ok / slow / undetectable / below_min_effect /
        no_expected_effect / unknown, with the numbers behind it."""
        out: dict[str, Any] = {"expected_effect": effect, "sd": self.sd,
                               "units_per_arm_per_window": self.units_per_arm_per_window,
                               "max_windows": screening.max_windows, "basis": self.basis}
        if effect is None:
            return {**out, "verdict": "no_expected_effect"}
        if abs(effect) < spec.min_effect:
            return {**out, "verdict": "below_min_effect"}
        needed = self.needed_per_arm(effect, screening)
        if needed is None or not self.units_per_arm_per_window:
            return {**out, "verdict": "unknown"}
        windows = math.ceil(needed / self.units_per_arm_per_window)
        out.update(needed_per_arm=needed, windows_needed=windows)
        if windows > screening.max_windows:
            return {**out, "verdict": "undetectable"}
        if windows > max(1, screening.max_windows // 2):
            return {**out, "verdict": "slow"}
        return {**out, "verdict": "ok"}


def _parse(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def estimate_capacity(ledger: Ledger, domain: str, spec: MetricSpec, screening: Screening
                      ) -> Capacity:
    """Spread of the metric and units produced per arm per window, from recorded data.

    Volume counts every unit produced in the domain over (at most) the last four
    windows, ending at the most recent unit, divided by the time that data covers,
    and splits it over two arms.
    """
    values: list[float] = []
    produced_by_unit: dict[str, datetime] = {}   # a unit in two experiments counts once
    lo, hi = spec.valid_range
    for exp in ledger.experiments(domain=domain):
        for obs in ledger.observations(exp.id):
            produced_by_unit[obs.unit_id] = _parse(obs.produced_at)
            v = obs.metrics.get(spec.name)
            if v is None or not math.isfinite(v):
                continue
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                continue
            values.append(float(v))

    sd: Optional[float] = None
    notes = []
    if len(values) >= MIN_VALUES:
        if screening.binary:
            p = statistics.fmean(values)
            sd = math.sqrt(p * (1 - p)) if 0 < p < 1 else None
            notes.append(f"base rate {p:.3g} from {len(values)} units")
        else:
            sd = statistics.stdev(values)
            notes.append(f"sd {sd:.3g} from {len(values)} units")
    else:
        notes.append(f"only {len(values)} usable values (need {MIN_VALUES})")

    produced = list(produced_by_unit.values())
    per_window = screening.units_per_arm_per_window
    if per_window is None and produced:
        latest = max(produced)
        span = 4 * screening.window
        recent = [t for t in produced if latest - t < span]
        if len(recent) >= MIN_VALUES:
            # divide by the time the data actually covers (at least one window), not by
            # four windows — a domain with one week of history isn't a quarter as busy
            covered = max((latest - min(recent)) / screening.window, 1.0)
            per_window = len(recent) / covered / 2
            notes.append(f"{len(recent)} units over {covered:.1f} windows")
        else:
            notes.append(f"only {len(recent)} units in the last 4 windows")
    elif per_window is not None:
        notes.append("volume given")
    return Capacity(sd=sd, units_per_arm_per_window=per_window, basis="; ".join(notes))
