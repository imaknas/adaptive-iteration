"""Where events are kept, and where "now" comes from."""
from datetime import datetime, timedelta, timezone

import pytest

from ordal import (
    Evaluator,
    Experiment,
    JsonlFile,
    Ledger,
    MemoryLog,
    MetricSpec,
    Observation,
    Variant,
    restart,
)
from ordal.domain import DomainConfig, config_for, save_config
from ordal.eventlog import LedgerReadOnlyError

T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
SPEC = MetricSpec("m", min_effect=3.0)


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, at=T0):
        self.at = at

    def __call__(self):
        return self.at

    def advance(self, **kw):
        self.at += timedelta(**kw)


def run_script(ledger):
    exp = Experiment(id="e1", domain="d", variable="v", variant_a=Variant("a"),
                     variant_b=Variant("b"))
    ledger.add_experiment(exp)
    ledger.start_experiment(exp.id)
    for i in range(6):
        for label, value in (("a", 50.0 + i), ("b", 70.0 + i)):
            ledger.record_observation(Observation(
                exp.id, label, f"{label}{i}", T0.isoformat(),
                (T0 + timedelta(days=4)).isoformat(), {"m": value}))
    return exp


def test_file_and_memory_logs_hold_identical_events(tmp_path):
    clock = Clock()
    on_disk = Ledger(JsonlFile(tmp_path / "l.jsonl"), clock=clock)
    in_memory = Ledger(MemoryLog(), clock=clock)
    run_script(on_disk)
    run_script(in_memory)
    assert on_disk.events == in_memory.events
    reopened = Ledger(JsonlFile(tmp_path / "l.jsonl"), clock=clock)
    assert reopened.events == on_disk.events


def test_every_timestamp_comes_from_the_clock():
    clock = Clock()
    ledger = Ledger(MemoryLog(), clock=clock)
    exp = run_script(ledger)
    assert {e["recorded_at"] for e in ledger.events} == {T0.isoformat()}
    assert ledger.experiment(exp.id).started == T0.isoformat()

    clock.advance(days=8)                     # evaluate() without `now` uses the clock
    d = Evaluator(ledger).evaluate(exp.id, SPEC)
    assert d.checkpoint == 1 and d.decided_at == clock().isoformat()


def test_settings_follow_the_clock_not_wall_time():
    """Configure, start, reconfigure, restart — with a clock we control, no backdating."""
    clock = Clock()
    ledger = Ledger(MemoryLog(), clock=clock)
    save_config(ledger, DomainConfig("d", MetricSpec("m", min_effect=5.0)))
    clock.advance(days=1)
    exp = Experiment(domain="d", variable="v", variant_a=Variant("a"), variant_b=Variant("b"))
    ledger.add_experiment(exp)
    ledger.start_experiment(exp.id)
    clock.advance(days=1)
    save_config(ledger, DomainConfig("d", MetricSpec("m", min_effect=1.0)))   # loosen it
    assert config_for(ledger, exp).metric.min_effect == 5.0      # running one keeps its bar
    clock.advance(days=1)
    new = restart(ledger, exp.id, "pipeline changed")
    assert config_for(ledger, new).metric.min_effect == 1.0      # restart takes current bar


def test_any_event_log_can_back_a_ledger():
    class Recording:
        read_only = False

        def __init__(self):
            self.appended = []

        def load(self):
            return []

        def append(self, event):
            self.appended.append(event)

        def describe(self):
            return "recording"

    log = Recording()
    ledger = Ledger(log, clock=Clock())
    run_script(ledger)
    assert len(log.appended) == len(ledger) and "recording" in repr(ledger)


def test_paths_and_none_still_work(tmp_path):
    assert isinstance(Ledger(tmp_path / "x.jsonl").log, JsonlFile)
    assert isinstance(Ledger(str(tmp_path / "y.jsonl")).log, JsonlFile)
    assert isinstance(Ledger(None).log, MemoryLog)
    assert isinstance(Ledger().log, MemoryLog)


def test_v1_file_is_read_only(tmp_path):
    p = tmp_path / "old.json"
    p.write_text('[{"domain": "d", "experiment_id": "x", "variable": "v", "variant": "a",'
                 ' "metric_values": {"m": 1}, "winner": true, "timestamp": "2026-08-01"}]')
    ledger = Ledger(p)
    assert ledger.read_only and len(ledger.legacy_records("d")) == 1
    with pytest.raises(LedgerReadOnlyError):
        ledger.add_experiment(Experiment(domain="d", variable="v"))
    assert len(ledger) == 1                    # nothing half-written in memory either
