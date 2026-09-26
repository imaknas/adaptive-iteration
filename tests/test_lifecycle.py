"""abandon / restart, the start-approval hold, and shadow mode."""
import shutil
from datetime import timedelta

import pytest
from test_loop import SPEC, T0, Fixed, Ideas, Pipeline, configured, prop, run, started

from adaptive_iteration import (
    Evaluator,
    Ledger,
    Loop,
    MetricSpec,
    Observation,
    abandon,
    assign,
    build_evidence,
    restart,
    service,
)
from adaptive_iteration.core.domain import DomainConfig, config_for, save_config
from adaptive_iteration.core.hypothesis import HypothesisEngine

# ── abandon ───────────────────────────────────────────────────────────────────

def test_abandoned_experiment_is_out_of_everything(tmp_path):
    ledger = configured(tmp_path)
    exp = started(ledger, "hook")
    abandon(ledger, exp.id, "prompt changed")
    assert not ledger.is_open(exp.id)
    with pytest.raises(ValueError, match="closed"):
        assign(ledger, exp.id, "u1")
    with pytest.raises(ValueError, match="abandoned"):
        ledger.record_observation(Observation(exp.id, "a", "u1", T0.isoformat(),
                                              T0.isoformat(), {"m": 1.0}))
    with pytest.raises(ValueError, match="no verdict"):
        Evaluator(ledger).evaluate(exp.id, SPEC)
    # the variable is free again
    [r] = HypothesisEngine(ledger, Fixed([prop("hook", 20.0)])).generate("d", SPEC)
    assert r.accepted
    ev = build_evidence(ledger, "d", SPEC)
    assert {v.variable: v.status for v in ev.variables}["hook"] == "abandoned"
    assert ev.proposers["unknown"]["abandoned"] == 1


def test_abandon_guards(tmp_path):
    ledger = configured(tmp_path)
    exp = started(ledger)
    with pytest.raises(ValueError, match="reason"):
        abandon(ledger, exp.id, "  ")
    abandon(ledger, exp.id, "wrong setup")
    with pytest.raises(ValueError, match="already abandoned"):
        abandon(ledger, exp.id, "again")


# ── restart ───────────────────────────────────────────────────────────────────

def test_restart_rebuilds_the_experiment_under_current_settings(tmp_path):
    ledger = configured(tmp_path)
    old = started(ledger, "hook")
    # the pipeline changes and the domain is reconfigured after the experiment began
    save_config(ledger, DomainConfig(domain="d", metric=MetricSpec("m", min_effect=9.0)))
    assert config_for(ledger, old).metric.min_effect == 5.0     # old keeps its settings
    new = restart(ledger, old.id, "config v7", at=(T0 + timedelta(days=30)).isoformat())
    assert new.id != old.id and new.restart_of == old.id
    assert (new.variable, new.variant_a, new.variant_b) == (old.variable, old.variant_a,
                                                           old.variant_b)
    assert new.started.startswith("2026-10-07")
    assert ledger.abandoned(old.id)["reason"] == "config v7"
    assert config_for(ledger, new).metric.min_effect == 9.0      # judged under new settings


def test_service_abandon_restart_and_atomic_records(tmp_path):
    L = str(tmp_path / "s.jsonl")
    service.configure(L, "d", "m", 5.0)
    service.register_variable(L, "d", "v")
    exp_id = service.accept_proposal(L, "d", {"variable": "v", "variant_a": "a",
                                              "variant_b": "b"})["experiment"]["id"]
    out = service.restart_experiment(L, exp_id, "model upgraded")
    assert out["abandoned"]["status"] == "abandoned"
    new_id = out["experiment"]["id"]
    assert out["experiment"]["restart_of"] == exp_id and out["experiment"]["status"] == "running"
    rows = [{"experiment_id": new_id, "variant": "a", "unit_id": "ok",
             "produced_at": T0.isoformat(), "metrics": {"m": 1}},
            {"experiment_id": exp_id, "variant": "a", "unit_id": "late",
             "produced_at": T0.isoformat(), "metrics": {"m": 1}}]
    with pytest.raises(service.ServiceError, match="abandoned"):
        service.record_observations(L, rows)
    assert service.status(L, "d")["d"]["experiments"][1]["observations"] == 0  # nothing written


# ── start approval ───────────────────────────────────────────────────────────

def test_start_approval_holds_the_slot_until_decided(tmp_path):
    ledger = configured(tmp_path)
    pipe = Pipeline(truth={"hook": 15.0}, seed=1)
    loop = Loop(ledger, "d", collect=pipe.collect, apply=pipe.apply, proposer=Ideas(),
                require_start_approval=True)
    r1 = loop.tick(now=pipe.now)
    assert r1.started == [] and [w["variable"] for w in r1.awaiting_start] == ["hook"]
    r2 = loop.tick(now=pipe.now)                        # slot is held: nothing new added
    assert len(ledger.experiments("d")) == 1 and len(r2.awaiting_start) == 1
    assert loop.variant_for("u0") == []                 # nothing running yet

    loop.reject_start(r1.awaiting_start[0]["experiment_id"], "not this week")
    r3 = loop.tick(now=pipe.now)                        # slot freed: next idea offered
    assert [w["variable"] for w in r3.awaiting_start] == ["thumbnail"]

    loop.approve_start(r3.awaiting_start[0]["experiment_id"], at=pipe.now)
    assert [a.variable for a in loop.variant_for("u0")] == ["thumbnail"]
    with pytest.raises(ValueError, match="not waiting"):
        loop.approve_start(r3.awaiting_start[0]["experiment_id"])


def test_loop_restart_mid_run_then_concludes(tmp_path):
    ledger = configured(tmp_path)
    pipe = Pipeline(truth={"hook": 15.0}, seed=2)
    ideas = Ideas()
    ideas.queue = ideas.queue[:1]
    loop = Loop(ledger, "d", collect=pipe.collect, apply=pipe.apply, proposer=ideas)
    loop.tick(now=pipe.now)
    pipe.produce(loop, 30)
    [old] = loop.running()
    new = loop.restart(old.id, "prompt v2", at=pipe.now)
    assert [e.id for e in loop.running()] == [new.id]
    run(loop, pipe, weeks=4)
    assert pipe.config.get("hook") == "b"
    assert ledger.final_decision(new.id) is not None and ledger.applied(old.id) is None


# ── shadow mode ───────────────────────────────────────────────────────────────

def test_shadow_loop_on_a_copy_leaves_the_real_ledger_alone(tmp_path):
    ledger = configured(tmp_path, max_windows=1)       # one checkpoint: a verdict is final
    pipe = Pipeline(truth={"hook": 25.0}, seed=3)       # a clear effect, so the test isn't
    live = Loop(ledger, "d", collect=pipe.collect,      # about statistical luck
                apply=pipe.apply, proposer=Ideas())
    live.tick(now=pipe.now)
    pipe.produce(live, 120)
    pipe.now += timedelta(days=7)
    real_path = tmp_path / "l.jsonl"
    before = real_path.read_text()

    shadow_path = tmp_path / "shadow.jsonl"
    shutil.copy(real_path, shadow_path)
    would_apply = []
    shadow = Loop(Ledger(shadow_path), "d", collect=pipe.collect,
                  apply=lambda e, v, d: would_apply.append((e.variable, v)),
                  proposer=Fixed([]))
    report = shadow.tick(now=pipe.now)
    assert report.verdicts and report.started == []
    assert real_path.read_text() == before          # real ledger untouched
    assert would_apply == [("hook", "b")]           # what it would have done
