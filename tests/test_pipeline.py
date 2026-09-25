"""Ledger, Evaluator, registry, evidence, HypothesisEngine and migration, end to end."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from adaptive_iteration import (
    Evaluator,
    Experiment,
    HypothesisEngine,
    Ledger,
    MetricSpec,
    Observation,
    Outcome,
    Proposal,
    VariableDef,
    VariableRegistry,
    Variant,
    build_evidence,
)
from adaptive_iteration.core.ledger import LedgerReadOnlyError
from adaptive_iteration.migrate import v1_to_v2

SPEC = MetricSpec(name="avg_view_pct", min_effect=3.0)
T0 = datetime(2026, 9, 1, tzinfo=timezone.utc)


def iso(dt):
    return dt.isoformat()


def make_experiment(ledger, variable="hook_style", domain="yt"):
    exp = Experiment(domain=domain, variable=variable,
                     variant_a=Variant("question"), variant_b=Variant("scenario"))
    ledger.add_experiment(exp)
    ledger.start_experiment(exp.id, at=iso(T0))
    return exp


def add_obs(ledger, exp, variant, values, produced=T0, observed=None):
    observed = observed or produced + timedelta(days=4)
    for i, v in enumerate(values):
        ledger.record_observation(Observation(
            experiment_id=exp.id, variant=variant, unit_id=f"{variant}-{i}",
            produced_at=iso(produced), observed_at=iso(observed),
            metrics={"avg_view_pct": v}))


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "ledger.jsonl")


# ── Ledger ─────────────────────────────────────────────────────────────────────

def test_ledger_roundtrip_and_latest_observation_wins(ledger, tmp_path):
    exp = make_experiment(ledger)
    add_obs(ledger, exp, "question", [40.0], observed=T0 + timedelta(days=1))
    add_obs(ledger, exp, "question", [55.0], observed=T0 + timedelta(days=5))
    reopened = Ledger(tmp_path / "ledger.jsonl")
    assert reopened.experiment(exp.id).started == iso(T0)
    [obs] = reopened.observations(exp.id)
    assert obs.metrics["avg_view_pct"] == 55.0


def test_observation_for_unknown_experiment_rejected(ledger):
    with pytest.raises(KeyError):
        ledger.record_observation(Observation("nope", "a", "u", iso(T0), iso(T0), {}))


# ── Evaluator ──────────────────────────────────────────────────────────────────

def test_before_first_checkpoint_nothing_recorded(ledger):
    exp = make_experiment(ledger)
    add_obs(ledger, exp, "question", [50] * 10)
    add_obs(ledger, exp, "scenario", [70] * 10)
    d = Evaluator(ledger).evaluate(exp.id, SPEC, now=T0 + timedelta(days=6))
    assert d.outcome is Outcome.INSUFFICIENT and d.checkpoint == 0
    assert ledger.decisions(exp.id) == []


def test_checkpoint_decision_recorded_once_and_reused(ledger):
    exp = make_experiment(ledger)
    add_obs(ledger, exp, "question", [50, 52, 48, 51, 49, 50])
    add_obs(ledger, exp, "scenario", [70, 72, 68, 71, 69, 70])
    ev = Evaluator(ledger)
    d1 = ev.evaluate(exp.id, SPEC, now=T0 + timedelta(days=8))
    d2 = ev.evaluate(exp.id, SPEC, now=T0 + timedelta(days=30))
    assert d1.outcome is Outcome.B_BETTER and d1.checkpoint == 1
    assert d2 == d1                       # final decision closes the experiment
    assert len(ledger.decisions(exp.id)) == 1


def test_missing_immature_and_out_of_range_are_excluded_not_zero(ledger):
    exp = make_experiment(ledger)
    spec = MetricSpec(name="avg_view_pct", min_effect=3.0, valid_range=(0.0, 500.0))
    add_obs(ledger, exp, "question", [50, 51, 49, 50, 52])
    add_obs(ledger, exp, "scenario", [50, 51, 49, 50, 52])
    ledger.record_observation(Observation(exp.id, "scenario", "gap", iso(T0),
                                          iso(T0 + timedelta(days=4)), {"avg_view_pct": None}))
    ledger.record_observation(Observation(exp.id, "scenario", "fresh", iso(T0),
                                          iso(T0 + timedelta(hours=10)), {"avg_view_pct": 0.0}))
    ledger.record_observation(Observation(exp.id, "scenario", "weird", iso(T0),
                                          iso(T0 + timedelta(days=4)), {"avg_view_pct": -5.0}))
    d = Evaluator(ledger).evaluate(exp.id, spec, now=T0 + timedelta(days=8))
    assert d.n_b == 5
    assert d.excluded["missing"] == 1
    assert d.excluded["immature"] == 1
    assert d.excluded["out_of_range"] == 1


# ── Registry ───────────────────────────────────────────────────────────────────

def test_registry_merge_and_alias_chains(ledger):
    reg = VariableRegistry(ledger, "yt")
    for n in ("intro_visual_style", "opening_visual_style", "visual_hook"):
        reg.register(VariableDef(n))
    reg.merge("intro_visual_style", into="opening_visual_style")
    reg.merge("opening_visual_style", into="visual_hook")
    assert reg.resolve("intro_visual_style") == "visual_hook"
    assert [v.name for v in reg.variables()] == ["visual_hook"]
    assert set(reg.get("visual_hook").aliases) == {"intro_visual_style", "opening_visual_style"}
    with pytest.raises(ValueError):
        reg.register(VariableDef("intro_visual_style"))


def test_registry_is_scoped_per_domain(ledger):
    VariableRegistry(ledger, "yt").register(VariableDef("hook_style"))
    assert VariableRegistry(ledger, "ig").resolve("hook_style") is None


# ── HypothesisEngine ───────────────────────────────────────────────────────────

class StubProposer:
    def __init__(self, proposals):
        self.proposals = proposals
        self.seen = None

    def propose(self, evidence, n):
        self.seen = evidence
        return self.proposals[:n]


def prop(variable, new=None, execution=None):
    return Proposal(variable=variable, description="", variant_a=Variant("a"),
                    variant_b=Variant("b"),
                    new_variable=VariableDef(variable, execution=execution) if new else None)


def test_review_statuses(ledger):
    reg = VariableRegistry(ledger, "yt")
    reg.register(VariableDef("opening_visual_style", execution="first-frame template"))
    reg.register(VariableDef("hook_style"))
    make_experiment(ledger, variable="hook_style")           # open experiment
    engine = HypothesisEngine(ledger, StubProposer([
        prop("opening_visual_style"),                          # known
        prop("visual_opening_style", new=True),                # token-set duplicate → merged
        prop("subtitle_position", new=True),                   # new, no execution → flagged
        prop("music_bpm"),                                     # unregistered, no definition
        prop("hook_style"),                                    # already open
    ]))
    r = engine.generate("yt", SPEC, n=5)
    assert [x.status for x in r] == ["known", "rejected", "new", "rejected", "rejected"]
    # the merged duplicate is rejected because the batch already holds that variable
    assert "open experiment" in r[1].reason
    assert r[2].flags == ("needs_execution",)
    assert "not registered" in r[3].reason
    assert "open experiment" in r[4].reason


def test_merged_duplicate_maps_to_canonical(ledger):
    VariableRegistry(ledger, "yt").register(VariableDef("opening_visual_style"))
    engine = HypothesisEngine(ledger, StubProposer([prop("visual_opening_style", new=True)]))
    [r] = engine.generate("yt", SPEC)
    assert r.status == "merged" and r.variable == "opening_visual_style"


def test_accept_registers_new_variable_and_adds_experiment(ledger):
    engine = HypothesisEngine(ledger, StubProposer([prop("subtitle_position", new=True,
                                                         execution="ffmpeg overlay y")]))
    [r] = engine.generate("yt", SPEC)
    assert VariableRegistry(ledger, "yt").resolve("subtitle_position") is None   # not yet
    exp = engine.accept(r)
    assert VariableRegistry(ledger, "yt").resolve("subtitle_position") == "subtitle_position"
    assert ledger.experiment(exp.id).variable == "subtitle_position"


def test_proposer_receives_structured_evidence(ledger):
    exp = make_experiment(ledger)
    add_obs(ledger, exp, "question", [50, 52, 48, 51, 49])
    add_obs(ledger, exp, "scenario", [70, 72, 68, 71, 69])
    Evaluator(ledger).evaluate(exp.id, SPEC, now=T0 + timedelta(days=8))
    stub = StubProposer([])
    HypothesisEngine(ledger, stub).generate("yt", SPEC)
    [v] = stub.seen.variables
    assert v.variable == "hook_style" and v.status == "concluded" and v.best_variant == "scenario"
    assert "hook_style" in stub.seen.to_markdown()


# ── Migration ──────────────────────────────────────────────────────────────────

V1 = [
    {"domain": "yt", "experiment_id": "x1", "variable": "thumbnail_style",
     "variant": "minimal", "metric_values": {"avg_view_pct": 130.77}, "winner": True,
     "timestamp": "2026-08-22T06:00:00+00:00"},
    {"domain": "yt", "experiment_id": "x1", "variable": "thumbnail_style",
     "variant": "high_contrast", "metric_values": {"avg_view_pct": 72.2}, "winner": False,
     "timestamp": "2026-08-22T06:00:00+00:00"},
]


def test_v1_ledger_opens_read_only(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(json.dumps(V1))
    old = Ledger(p)
    assert old.read_only and len(old.legacy_records("yt")) == 2
    with pytest.raises(LedgerReadOnlyError):
        old.add_experiment(Experiment(domain="yt", variable="v"))


def test_migration_keeps_source_and_marks_legacy_unverified(tmp_path):
    src, dst = tmp_path / "old.json", tmp_path / "new.jsonl"
    src.write_text(json.dumps(V1))
    new = v1_to_v2(src, dst)
    assert json.loads(src.read_text()) == V1
    assert VariableRegistry(new, "yt").resolve("thumbnail_style") == "thumbnail_style"
    [v] = build_evidence(Ledger(dst), "yt", SPEC).variables
    assert v.status == "legacy_unverified" and v.best_variant is None
    with pytest.raises(FileExistsError):
        v1_to_v2(src, dst)
