"""Assignment, screening, proposer track record, and the Loop driver end to end."""
import random
from datetime import datetime, timedelta, timezone

import pytest

from adaptive_iteration import (
    Experiment,
    Ledger,
    Loop,
    MetricSpec,
    Observation,
    Outcome,
    Proposal,
    Screening,
    VariableDef,
    VariableRegistry,
    Variant,
    assign,
    build_evidence,
    service,
)
from adaptive_iteration.core.domain import DomainConfig, save_config
from adaptive_iteration.core.hypothesis import HypothesisEngine

T0 = datetime(2026, 9, 7, tzinfo=timezone.utc)
SPEC = MetricSpec(name="m", min_effect=5.0)


def configured(tmp_path, **kw):
    ledger = Ledger(tmp_path / "l.jsonl")
    save_config(ledger, DomainConfig(domain="d", metric=SPEC, **kw))
    return ledger


def started(ledger, variable="v", mode="interleaved"):
    exp = Experiment(domain="d", variable=variable, variant_a=Variant("a"),
                     variant_b=Variant("b"), mode=mode)
    ledger.add_experiment(exp)
    ledger.start_experiment(exp.id, at=T0.isoformat())
    return exp


# ── Assignment ────────────────────────────────────────────────────────────────

def test_assignment_is_balanced_within_strata_and_stable(tmp_path):
    ledger = configured(tmp_path)
    exp = started(ledger)
    labels = {s: [assign(ledger, exp.id, f"{s}{i}", s) for i in range(9)]
              for s in ("money", "other")}
    for s, got in labels.items():
        assert abs(got.count("a") - got.count("b")) <= 1, (s, got)
    assert assign(ledger, exp.id, "money0", "money") == labels["money"][0]


def test_assignment_refuses_paired_and_closed(tmp_path):
    ledger = configured(tmp_path)
    with pytest.raises(ValueError, match="paired"):
        assign(ledger, started(ledger, "p", mode="paired").id, "u1")
    exp = started(ledger, "c")
    ledger.mark_applied  # noqa: B018 - just checking the API exists
    from adaptive_iteration.core.decision import Decision
    ledger.record_decision(Decision(exp.id, Outcome.EQUIVALENT, "m", 1, 5, 5, {}, "r", {},
                                    "x", T0.isoformat()))
    with pytest.raises(ValueError, match="closed"):
        assign(ledger, exp.id, "u1")


def test_observation_contradicting_assignment_is_refused(tmp_path):
    ledger = configured(tmp_path)
    exp = started(ledger)
    got = assign(ledger, exp.id, "u1")
    other = "b" if got == "a" else "a"
    with pytest.raises(ValueError, match="was assigned"):
        ledger.record_observation(Observation(exp.id, other, "u1", T0.isoformat(),
                                              T0.isoformat(), {"m": 1.0}))
    ledger.record_observation(Observation(exp.id, got, "u1", T0.isoformat(),
                                          T0.isoformat(), {"m": 1.0}))


# ── Screening ─────────────────────────────────────────────────────────────────

class Fixed:
    def __init__(self, proposals, name="fixed"):
        self.proposals, self.name = proposals, name

    def propose(self, evidence, n):
        return self.proposals[:n]


def with_history(ledger, n=200, sd=12.0):
    """Give the domain a closed experiment's worth of data: n units over 4 weeks."""
    rng = random.Random(0)
    exp = started(ledger, "history")
    for i in range(n):
        t = T0 + timedelta(hours=i * 672 / n)
        label = assign(ledger, exp.id, f"h{i}")
        ledger.record_observation(Observation(exp.id, label, f"h{i}", t.isoformat(),
                                              (t + timedelta(days=4)).isoformat(),
                                              {"m": rng.gauss(60, sd)}))


def prop(var, effect):
    return Proposal(var, "", Variant("a"), Variant("b"), expected_effect=effect,
                    new_variable=VariableDef(var, execution="x"))


def test_screen_rejects_undetectable_and_trivial_and_flags_the_rest(tmp_path):
    ledger = configured(tmp_path)
    with_history(ledger)                               # sd≈12, 25 units/arm/week
    engine = HypothesisEngine(ledger, Fixed([
        prop("tiny", 2.0),      # below min_effect
        prop("small", 5.0),     # needs ~250/arm → ~10 weeks > 4
        prop("medium", 8.0),    # a few weeks → slow
        prop("big", 20.0),      # ok
        prop("vague", None),    # no estimate given
    ]), screening=Screening())
    r = {x.proposal.variable: x for x in engine.generate("d", SPEC, n=5)}
    assert r["tiny"].status == "rejected" and "below min_effect" in r["tiny"].reason
    assert r["small"].status == "rejected" and "windows" in r["small"].reason
    assert r["medium"].accepted and "slow" in r["medium"].flags
    assert r["big"].accepted and r["big"].screening["verdict"] == "ok"
    assert r["vague"].accepted and "no_expected_effect" in r["vague"].flags


def test_screen_says_unknown_without_data(tmp_path):
    ledger = configured(tmp_path)
    engine = HypothesisEngine(ledger, Fixed([prop("x", 10.0)]), screening=Screening())
    [r] = engine.generate("d", SPEC)
    assert r.accepted and "detectability_unknown" in r.flags


# ── Track record ──────────────────────────────────────────────────────────────

def test_track_record_per_proposer(tmp_path):
    ledger = configured(tmp_path)
    for name, var in (("llm", "x"), ("llm", "y"), ("human", "z")):
        engine = HypothesisEngine(ledger, Fixed([prop(var, 10.0)], name=name))
        [r] = engine.generate("d", SPEC)
        engine.accept(r)
    ev = build_evidence(ledger, "d", SPEC)
    assert ev.proposers["llm"]["proposed"] == 2 and ev.proposers["human"]["proposed"] == 1
    assert "Proposer track record" in ev.to_markdown()


# ── Loop, end to end ──────────────────────────────────────────────────────────

class Pipeline:
    """A fake content pipeline with known true effects per variable."""

    def __init__(self, truth, seed=0):
        self.truth = truth                 # variable -> true B − A effect
        self.rng = random.Random(seed)
        self.units = []                    # (unit_id, stratum, produced_at, {exp_id: label}, value)
        self.config = {}                   # variable -> variant put into effect
        self.now = T0
        self.fail_apply = False

    def produce(self, loop, n):
        for i in range(n):
            uid = f"u{len(self.units)}"
            stratum = "money" if self.rng.random() < 0.4 else "other"
            produced = self.now + timedelta(hours=i * 168 / n)
            value = self.rng.gauss(90 if stratum == "money" else 60, 12)
            labels = {}
            for a in loop.variant_for(uid, stratum):
                labels[a.experiment_id] = a.variant.label
                if a.variant.label == "b":
                    value += self.truth.get(a.variable, 0.0)
            self.units.append((uid, stratum, produced, labels, value))

    def collect(self, exp):
        return [Observation(exp.id, labels[exp.id], uid, produced.isoformat(),
                            min(self.now, produced + timedelta(days=4)).isoformat(),
                            {"m": value}, stratum=stratum)
                for uid, stratum, produced, labels, value in self.units
                if exp.id in labels and produced <= self.now]

    def apply(self, exp, variant, decision):
        if self.fail_apply:
            raise RuntimeError("deploy failed")
        self.config[exp.variable] = variant


class Ideas:
    name = "ideas"

    def __init__(self):
        self.queue = [prop("hook", 12.0), prop("thumbnail", 12.0), prop("cta", 12.0)]

    def propose(self, evidence, n):
        tried = {v.variable for v in evidence.variables if v.status != "untested"}
        return [p for p in self.queue if p.variable not in tried][:n]


def run(loop, pipe, weeks, per_week=60):
    reports = []
    for _ in range(weeks):
        pipe.produce(loop, per_week)
        pipe.now += timedelta(days=7)
        reports.append(loop.tick(now=pipe.now))
    return reports


def test_loop_finds_the_real_improvement_and_moves_on(tmp_path):
    ledger = configured(tmp_path)
    pipe = Pipeline(truth={"hook": 15.0, "thumbnail": 0.0, "cta": 0.0}, seed=4)
    loop = Loop(ledger, "d", collect=pipe.collect, apply=pipe.apply, proposer=Ideas())
    loop.tick(now=pipe.now)                              # starts the first experiment
    reports = run(loop, pipe, weeks=14)
    assert pipe.config.get("hook") == "b"                # real +15 found and adopted
    assert pipe.config.get("thumbnail") in (None, "a")   # no effect → never adopts B
    assert pipe.config.get("cta") in (None, "a")
    started_vars = [s["variable"] for r in reports for s in r["started"]] \
        if isinstance(reports[0], dict) else [s["variable"] for r in reports for s in r.started]
    assert "thumbnail" in started_vars                   # moved on by itself
    assert not any(r.errors for r in reports)
    exps = {e.variable: e for e in ledger.experiments("d")}
    assert exps["hook"].proposed_by == "ideas"
    assert ledger.final_decision(exps["hook"].id).rule_params.get("stratified") is True


def test_loop_holds_a_b_win_for_approval(tmp_path):
    ledger = configured(tmp_path)
    pipe = Pipeline(truth={"hook": 15.0}, seed=5)
    ideas = Ideas()
    ideas.queue = ideas.queue[:1]
    loop = Loop(ledger, "d", collect=pipe.collect, apply=pipe.apply, proposer=ideas,
                require_approval=True)
    loop.tick(now=pipe.now)
    reports = run(loop, pipe, weeks=6)
    assert "hook" not in pipe.config
    assert any(r.awaiting_approval for r in reports)
    [exp] = loop.pending_approval()
    loop.approve(exp.id)
    assert pipe.config["hook"] == "b" and ledger.applied(exp.id)["variant"] == "b"
    assert loop.pending_approval() == []


def test_failed_apply_is_reported_and_retried(tmp_path):
    ledger = configured(tmp_path)
    pipe = Pipeline(truth={"hook": 15.0}, seed=5)
    ideas = Ideas()
    ideas.queue = ideas.queue[:1]
    loop = Loop(ledger, "d", collect=pipe.collect, apply=pipe.apply, proposer=ideas)
    loop.tick(now=pipe.now)
    pipe.fail_apply = True
    reports = run(loop, pipe, weeks=6)
    assert any("deploy failed" in e for r in reports for e in r.errors)
    assert "hook" not in pipe.config
    pipe.fail_apply = False
    loop.tick(now=pipe.now)
    assert pipe.config["hook"] == "b"


def test_tick_is_idempotent(tmp_path):
    ledger = configured(tmp_path)
    pipe = Pipeline(truth={"hook": 15.0}, seed=6)
    loop = Loop(ledger, "d", collect=pipe.collect, apply=pipe.apply, proposer=Ideas())
    loop.tick(now=pipe.now)
    run(loop, pipe, weeks=3)
    before = len(ledger)
    again = loop.tick(now=pipe.now)
    assert len(ledger) == before
    assert again.collected and not any(again.collected.values())


def test_service_assign_pending_and_mark_applied(tmp_path):
    L = str(tmp_path / "s.jsonl")
    service.configure(L, "d", "m", 5.0)
    VariableRegistry(Ledger(tmp_path / "s.jsonl"), "d").register(VariableDef("v"))
    exp_id = service.accept_proposal(L, "d", {"variable": "v", "variant_a": "a",
                                              "variant_b": "b", "expected_effect": 10,
                                              "proposed_by": "agent"},
                                     started_at=T0.isoformat())["experiment"]["id"]
    got = service.assign_variant(L, exp_id, "u1", "money")
    assert got["label"] in ("a", "b")
    wrong = "b" if got["label"] == "a" else "a"
    with pytest.raises(service.ServiceError, match="was assigned"):
        service.record_observations(L, [{"experiment_id": exp_id, "variant": wrong,
                                         "unit_id": "u1", "produced_at": T0.isoformat(),
                                         "metrics": {"m": 1}}])
    assert service.pending(L, "d") == []
    with pytest.raises(service.ServiceError, match="not a closed"):
        service.mark_applied(L, exp_id)
    ev = service.evidence(L, "d")
    assert ev["proposers"]["agent"]["proposed"] == 1
