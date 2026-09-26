"""Deterministic experiment ids in tests.

Experiment ids come from uuid4, and assignment breaks ties with a hash of the
experiment id, so random ids would make every simulated experiment see different
data from run to run — tests would pass or fail by luck. Each test gets the same
sequence of ids instead.
"""
import itertools
import uuid

import pytest


@pytest.fixture(autouse=True)
def deterministic_ids(monkeypatch):
    counter = itertools.count(1)
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(counter) << 96))  # ids use hex[:8]
