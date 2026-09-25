import random

import pytest

from adaptive_iteration.core.decision import (
    DecisionContext,
    Outcome,
    Sample,
    WelchIntervalRule,
)
from adaptive_iteration.core.metrics import MetricSpec

SPEC = MetricSpec(name="m", min_effect=2.0)
RULE = WelchIntervalRule()


def ctx(checkpoint=1, max_checkpoints=4, paired=False):
    return DecisionContext("e", paired=paired, checkpoint=checkpoint,
                           max_checkpoints=max_checkpoints)


def sample(values, pairs=None):
    return Sample(tuple(values), tuple(f"u{i}" for i in range(len(values))),
                  tuple(pairs) if pairs else tuple(None for _ in values))


def normal(rng, mean, sd, n):
    return [rng.gauss(mean, sd) for _ in range(n)]


def test_clear_winner_b():
    rng = random.Random(1)
    r = RULE.decide(sample(normal(rng, 50, 3, 30)), sample(normal(rng, 60, 3, 30)), SPEC, ctx())
    assert r.outcome is Outcome.B_BETTER
    assert r.effect == pytest.approx(10, abs=2)


def test_direction_respects_lower_is_better():
    rng = random.Random(2)
    spec = MetricSpec(name="m", min_effect=2.0, higher_is_better=False)
    r = RULE.decide(sample(normal(rng, 50, 3, 30)), sample(normal(rng, 60, 3, 30)), spec, ctx())
    assert r.outcome is Outcome.A_BETTER
    assert r.effect < 0


def test_equivalent_when_interval_inside_rope():
    rng = random.Random(3)
    r = RULE.decide(sample(normal(rng, 50, 1, 200)), sample(normal(rng, 50, 1, 200)), SPEC, ctx())
    assert r.outcome is Outcome.EQUIVALENT


def test_noisy_small_sample_is_insufficient_not_a_winner():
    # the v0.1 failure mode: 3 videos per arm, a 10-point gap in averages
    a, b = sample([60, 85, 70]), sample([80, 95, 72])
    r = WelchIntervalRule(min_n=2).decide(a, b, SPEC, ctx())
    assert r.outcome is Outcome.INSUFFICIENT
    assert r.needed_n and r.needed_n > 0


def test_below_min_n():
    r = RULE.decide(sample([1, 2, 3]), sample([4, 5, 6, 7, 8]), SPEC, ctx())
    assert r.outcome is Outcome.INSUFFICIENT
    assert "min_n" in r.reason


def test_last_checkpoint_closes_as_no_detectable_diff():
    a, b = sample([60, 85, 70, 66, 90]), sample([80, 95, 72, 61, 88])
    r = RULE.decide(a, b, SPEC, ctx(checkpoint=4))
    assert r.outcome is Outcome.NO_DETECTABLE_DIFF


def test_paired_uses_pair_differences():
    rng = random.Random(4)
    base = normal(rng, 50, 20, 12)          # large between-topic variation
    a = sample(base, pairs=[f"p{i}" for i in range(12)])
    b = sample([x + 5 + rng.gauss(0, 0.5) for x in base], pairs=[f"p{i}" for i in range(12)])
    assert RULE.decide(a, b, SPEC, ctx(paired=True)).outcome is Outcome.B_BETTER
    # the same data judged as unpaired is swamped by between-topic variance
    assert RULE.decide(a, b, SPEC, ctx(paired=False)).outcome is Outcome.INSUFFICIENT


def test_min_n_validation():
    with pytest.raises(ValueError):
        WelchIntervalRule(min_n=1)


def _simulate(effect, sims, per_window=10, windows=4, sd=10.0, seed=0):
    """Run the rule at each weekly checkpoint on accumulating data, as the Evaluator would."""
    rng = random.Random(seed)
    spec = MetricSpec(name="m", min_effect=2.0)
    outcomes = []
    for _ in range(sims):
        a, b = [], []
        for k in range(1, windows + 1):
            a += normal(rng, 50, sd, per_window)
            b += normal(rng, 50 + effect, sd, per_window)
            r = RULE.decide(sample(a), sample(b), spec, ctx(checkpoint=k, max_checkpoints=windows))
            if r.outcome.is_final:
                break
        outcomes.append(r.outcome)
    return outcomes


def test_false_positive_rate_with_repeated_checkpoints_stays_below_alpha():
    outcomes = _simulate(effect=0.0, sims=2000)
    wrong = sum(o in (Outcome.A_BETTER, Outcome.B_BETTER) for o in outcomes) / len(outcomes)
    assert wrong <= 0.05


def test_power_for_a_real_effect():
    # a 1-SD effect with 10 per arm per week: the rule must show B beats A by more than
    # min_effect (not merely "B is higher"), which lands around 85–90% by week 4
    outcomes = _simulate(effect=10.0, sims=500, seed=1)
    assert sum(o is Outcome.B_BETTER for o in outcomes) / len(outcomes) > 0.8
