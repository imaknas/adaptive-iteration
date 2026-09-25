import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from adaptive_iteration import (
    Evaluator,
    Experiment,
    Ledger,
    MetricSpec,
    Observation,
    Outcome,
    Variant,
    WelchIntervalRule,
    build_evidence,
)
from adaptive_iteration.core.decision import DecisionContext, RuleResult, Sample
from adaptive_iteration.replay import calibrate, gate, replay

SPEC = MetricSpec(name="m", min_effect=5.0)
# skewed, heavy-ish pool resembling real per-unit metrics
_rng = random.Random(42)
POOL = [max(1.0, _rng.lognormvariate(4.4, 0.3)) for _ in range(150)]


@dataclass(frozen=True)
class V01Rule:
    """The v0.1 short-video rule: ≥3 per arm, averages differ by more than weak_diff."""
    weak_diff: float = 5.0
    name: str = "v01_mean_diff"

    def decide(self, a: Sample, b: Sample, spec: MetricSpec, ctx: DecisionContext) -> RuleResult:
        if min(len(a), len(b)) < 3:
            return RuleResult(Outcome.INSUFFICIENT, "n < 3")
        diff = statistics.fmean(b.values) - statistics.fmean(a.values)
        if abs(diff) < self.weak_diff:
            return RuleResult(Outcome.NO_DETECTABLE_DIFF, "small diff")
        return RuleResult(Outcome.B_BETTER if diff > 0 else Outcome.A_BETTER, "mean diff")


def test_calibrate_false_positive_rate_on_skewed_pool():
    for sup in ("significance", "margin"):
        c = calibrate(POOL, WelchIntervalRule(superiority=sup), SPEC, effect=0,
                      per_window=10, sims=1000, seed=3)
        assert c.wrong_direction <= 0.06, (sup, c)
        assert c.mean_units_per_arm <= 40


def test_significance_detects_more_than_margin_at_same_effect():
    sig = calibrate(POOL, WelchIntervalRule(superiority="significance"), SPEC, effect=10,
                    per_window=20, sims=600, seed=4)
    mar = calibrate(POOL, WelchIntervalRule(superiority="margin"), SPEC, effect=10,
                    per_window=20, sims=600, seed=4)
    assert sig.detected > mar.detected


def test_gate_rejects_the_v01_rule_and_adopts_significance():
    incumbent = WelchIntervalRule(superiority="margin")
    bad = gate(V01Rule(), incumbent, POOL, SPEC, effects=(10,), per_window=5, sims=400)
    assert not bad.adopt
    assert any("false-positive" in r for r in bad.reasons)
    good = gate(WelchIntervalRule(superiority="significance"), incumbent, POOL, SPEC,
                effects=(10, 20), per_window=20, sims=400)
    assert good.adopt, good.reasons


def test_replay_only_sees_units_mature_by_each_checkpoint():
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    exp = Experiment(domain="d", variable="v", variant_a=Variant("a"), variant_b=Variant("b"),
                     started=start.isoformat())
    obs = []
    for day in range(0, 28):
        for label, mean in (("a", 50.0), ("b", 70.0)):
            produced = start + timedelta(days=day)
            obs.append(Observation("x", label, f"{label}{day}", produced.isoformat(),
                                   (start + timedelta(days=60)).isoformat(),
                                   {"m": mean + (day % 3)}))
    decisions = replay(exp, obs, SPEC)
    # week 1: produced ≤ day 4 is mature by day 7 → 5 per arm
    assert decisions[0].n_a == 5 and decisions[0].checkpoint == 1
    assert decisions[-1].outcome is Outcome.B_BETTER


def test_evidence_reports_cost_and_needed_n(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    spec = MetricSpec(name="m", min_effect=3.0)
    for var, (ma, mb) in {"x": (50, 70), "y": (50, 51)}.items():
        exp = Experiment(domain="d", variable=var, variant_a=Variant("a"), variant_b=Variant("b"))
        ledger.add_experiment(exp)
        ledger.start_experiment(exp.id, at=t0.isoformat())
        rng = random.Random(var)
        for i in range(8):
            for label, mean in (("a", ma), ("b", mb)):
                ledger.record_observation(Observation(
                    exp.id, label, f"{label}{i}", t0.isoformat(),
                    (t0 + timedelta(days=4)).isoformat(), {"m": rng.gauss(mean, 10)}))
        Evaluator(ledger).evaluate(exp.id, spec, now=t0 + timedelta(days=8))
    ev = build_evidence(ledger, "d", spec)
    by = {v.variable: v for v in ev.variables}
    assert by["x"].status == "concluded"
    assert by["y"].status == "open" and by["y"].needed_n and by["y"].needed_n > 0
    assert ev.cost["decisive"] == 1 and ev.cost["units_per_decisive"] == 16
    assert "Cost of judging" in ev.to_markdown()


def test_replay_keeps_strata():
    # regression: replay() rebuilt observations without their stratum, so a
    # stratified rule silently ran unstratified during replay
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    exp = Experiment(domain="d", variable="v", variant_a=Variant("a"), variant_b=Variant("b"),
                     started=start.isoformat())
    obs = [Observation("x", label, f"{label}{i}", (start + timedelta(hours=i)).isoformat(),
                       (start + timedelta(days=60)).isoformat(), {"m": 50.0 + i % 7},
                       stratum="money" if i % 2 else "other")
           for i in range(40) for label in ("a", "b")]
    [d, *_] = replay(exp, obs, SPEC)
    assert d.rule_params["stratified"] is True
