"""ProportionIntervalRule and stratified WelchIntervalRule."""
import random

import pytest

from adaptive_iteration import MetricSpec, Outcome, ProportionIntervalRule, WelchIntervalRule
from adaptive_iteration.core.decision import DecisionContext, Sample
from adaptive_iteration.replay import calibrate

REPLY = MetricSpec(name="replied", min_effect=0.10, valid_range=(0.0, 1.0))
PROP = ProportionIntervalRule()


def ctx(checkpoint=1, max_checkpoints=4, paired=False):
    return DecisionContext("e", paired=paired, checkpoint=checkpoint,
                           max_checkpoints=max_checkpoints)


def bern(k, n):
    return Sample(tuple([1.0] * k + [0.0] * (n - k)))


# ── ProportionIntervalRule ─────────────────────────────────────────────────────

def test_zero_events_gives_wide_interval_not_equivalence():
    r = PROP.decide(bern(0, 8), bern(0, 8), REPLY, ctx())
    assert r.outcome is Outcome.INSUFFICIENT
    lo, hi = r.interval
    assert lo < -0.2 and hi > 0.2


def test_clear_difference_in_proportions():
    r = PROP.decide(bern(20, 200), bern(80, 200), REPLY, ctx())
    assert r.outcome is Outcome.B_BETTER
    assert r.effect == pytest.approx(0.30)


def test_lower_is_better_flips_direction():
    spec = MetricSpec(name="bounced", min_effect=0.10, higher_is_better=False)
    r = PROP.decide(bern(20, 200), bern(80, 200), spec, ctx())
    assert r.outcome is Outcome.A_BETTER and r.effect < 0


def test_rejects_non_binary_and_delegates_paired():
    with pytest.raises(ValueError):
        PROP.decide(Sample((0.0, 0.5, 1.0, 1.0, 0.0)), bern(2, 5), REPLY, ctx())
    ids = tuple(str(i) for i in range(5))
    a = Sample((1.0, 0.0, 1.0, 0.0, 0.0), pair_ids=ids)
    b = Sample((1.0, 1.0, 1.0, 0.0, 1.0), pair_ids=ids)
    r = PROP.decide(a, b, REPLY, ctx(paired=True))
    assert r.params["pairs"] == 5               # handled by PairedProportionRule


def test_newcombe_matches_published_example():
    # Newcombe (1998), example (e): 56/70 vs 48/80 → 95% CI for difference 0.0524 to 0.3339
    rule = ProportionIntervalRule(alpha=0.05, min_n=2)
    r = rule.decide(bern(48, 80), bern(56, 70), MetricSpec("p", min_effect=0.01),
                    ctx(max_checkpoints=1))
    assert r.interval[0] == pytest.approx(0.0524, abs=5e-4)
    assert r.interval[1] == pytest.approx(0.3339, abs=5e-4)


def test_proportion_rule_false_positive_rate_with_rare_events():
    pool = [1.0] * 10 + [0.0] * 90                  # 10% base rate
    c = calibrate(pool, PROP, REPLY, effect=0, per_window=10, sims=1000, seed=2)
    assert c.wrong_direction <= 0.05


def test_proportion_rule_power_via_pool_b():
    pool_a = [1.0] * 10 + [0.0] * 90
    pool_b = [1.0] * 40 + [0.0] * 60                # +30 points
    c = calibrate(pool_a, PROP, REPLY, effect=0.30, pool_b=pool_b, per_window=15,
                  sims=500, seed=3)
    assert c.detected > 0.9


# ── Stratified WelchIntervalRule ───────────────────────────────────────────────

def stratified_samples(rng, effect, n=80, skew=0.8):
    """Stratum 'money' averages 40 points higher than 'other'. Arm A is mostly
    'other', arm B mostly 'money' — the confound a topic-skewed pipeline creates."""
    def arm(p_money, shift):
        vals, strata = [], []
        for _ in range(n):
            st = "money" if rng.random() < p_money else "other"
            base = 100.0 if st == "money" else 60.0
            vals.append(rng.gauss(base + shift, 10))
            strata.append(st)
        return Sample(tuple(vals), strata=tuple(strata))
    return arm(1 - skew, 0.0), arm(skew, effect)


def test_stratification_removes_topic_mix_confound():
    spec = MetricSpec(name="m", min_effect=3.0)
    rng = random.Random(5)
    a, b = stratified_samples(rng, effect=0.0)
    naive = WelchIntervalRule(stratify=False).decide(a, b, spec, ctx())
    strat = WelchIntervalRule().decide(a, b, spec, ctx())
    assert naive.outcome is Outcome.B_BETTER            # fooled by the mix
    assert strat.outcome is not Outcome.B_BETTER
    assert abs(strat.effect) < 3.0
    assert strat.params["stratified"] is True


def test_stratification_recovers_true_effect():
    spec = MetricSpec(name="m", min_effect=3.0)
    rng = random.Random(6)
    a, b = stratified_samples(rng, effect=8.0, n=120)
    r = WelchIntervalRule().decide(a, b, spec, ctx())
    assert r.outcome is Outcome.B_BETTER
    assert r.effect == pytest.approx(8.0, abs=3.0)


def test_strata_missing_an_arm_are_dropped_and_reported():
    spec = MetricSpec(name="m", min_effect=3.0)
    a = Sample((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 50.0), strata=("x",) * 6 + ("only_a",))
    b = Sample((1.5, 2.5, 3.5, 4.5, 5.5, 6.5), strata=("x",) * 6)
    r = WelchIntervalRule().decide(a, b, spec, ctx())
    assert r.params["dropped_units"] == 1
    assert "left out" in r.reason


def test_without_strata_behaviour_is_unchanged():
    spec = MetricSpec(name="m", min_effect=3.0)
    rng = random.Random(7)
    a = Sample(tuple(rng.gauss(50, 5) for _ in range(30)))
    b = Sample(tuple(rng.gauss(60, 5) for _ in range(30)))
    r1 = WelchIntervalRule().decide(a, b, spec, ctx())
    r2 = WelchIntervalRule(stratify=False).decide(a, b, spec, ctx())
    assert r1 == r2
