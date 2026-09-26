"""clock.py — Where "now" comes from.

Everything that needs the current time asks a Clock, so a test or a replay can
supply a fixed or simulated one. Programs use system_clock.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

Clock = Callable[[], datetime]


def system_clock() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(ts: str) -> datetime:
    """An ISO-8601 timestamp; one without a timezone is taken as UTC."""
    dt = datetime.fromisoformat(ts)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
