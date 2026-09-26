"""PairedProportionRule: paired 0/1 outcomes (Newcombe 1998, method 10)."""
import random

import pytest

from adaptive_iteration import (
    DecisionContext,
    MetricSpec,
    Outcome,
    PairedProportionRule,
    Sample,
)

RULE = PairedProportionRule()
SPEC = MetricSpec(name="recalled", min_effect=0.10)
ONE_LOOK = DecisionContext("batch", paired=True, checkpoint=1, max_checkpoints=1)


def paired(cells, start=0):
    """cells = (both, only_b, only_a, neither) → Samples A and B matched by pair_id."""
    both, only_b, only_a, neither = cells
    a = [1] * both + [0] * only_b + [1] * only_a + [0] * neither
    b = [1] * both + [1] * only_b + [0] * only_a + [0] * neither
    ids = tuple(f"p{start + i}" for i in range(len(a)))
    return (Sample(tuple(map(float, a)), pair_ids=ids),
            Sample(tuple(map(float, b)), pair_ids=ids))


def test_matches_newcombe_1998_reference():
    # Newcombe (1998) p.2641, row 2: n=50, both=20, only_b=12, only_a=2, neither=16
    # → difference 0.2, 95% CI 0.0562 to 0.3292 (method 10)
    a, b = paired((20, 12, 2, 16))
    r = RULE.decide(a, b, MetricSpec("p", min_effect=0.01), ONE_LOOK)
    assert r.effect == pytest.approx(0.2)
    assert r.interval[0] == pytest.approx(0.0562, abs=5e-5)
    assert r.interval[1] == pytest.approx(0.3292, abs=5e-5)
    assert r.outcome is Outcome.B_BETTER


def test_zero_discordant_pairs_at_small_n_is_not_equivalent():
    for cells in ((8, 0, 0, 0), (0, 0, 0, 8), (4, 0, 0, 4)):
        r = RULE.decide(*paired(cells), SPEC, ONE_LOOK)
        assert r.outcome is not Outcome.EQUIVALENT, cells
        lo, hi = r.interval
        assert lo < -SPEC.min_effect and hi > SPEC.min_effect


def test_all_concordant_interval_is_finite_and_shrinks_with_n():
    widths = []
    for n in (8, 40, 400):
        lo, hi = RULE.decide(*paired((n // 2, 0, 0, n - n // 2)), SPEC, ONE_LOOK).interval
        assert lo < 0 < hi and hi - lo < 2
        widths.append(hi - lo)
    assert widths[0] > widths[1] > widths[2]
    # with enough agreeing pairs, "no meaningful difference" is a fair conclusion
    big = RULE.decide(*paired((200, 0, 0, 200)), SPEC, ONE_LOOK)
    assert big.outcome is Outcome.EQUIVALENT


def test_unpaired_units_are_excluded_and_reported():
    a, b = paired((10, 5, 1, 4))
    a = Sample(a.values + (1.0, 1.0), pair_ids=a.pair_ids + ("lonely1", "lonely2"))
    r = RULE.decide(a, b, SPEC, ONE_LOOK)
    assert r.params["pairs"] == 20 and r.params["unpaired_units"] == 2
    assert "unpaired" in r.reason


def test_rejects_non_binary_and_duplicate_pair_ids():
    a, b = paired((3, 1, 1, 3))
    with pytest.raises(ValueError, match="0/1"):
        RULE.decide(Sample((0.5,) + a.values[1:], pair_ids=a.pair_ids), b, SPEC, ONE_LOOK)
    with pytest.raises(ValueError, match="twice"):
        RULE.decide(Sample(a.values, pair_ids=("x",) * len(a.values)), b, SPEC, ONE_LOOK)


def test_lower_is_better_flips_direction():
    spec = MetricSpec(name="errors", min_effect=0.10, higher_is_better=False)
    r = RULE.decide(*paired((20, 12, 2, 16)), spec, ONE_LOOK)
    assert r.effect == pytest.approx(-0.2) and r.outcome is Outcome.A_BETTER


def test_needed_pairs_estimate():
    r = RULE.decide(*paired((3, 2, 1, 2)), SPEC, ONE_LOOK)
    assert r.outcome is Outcome.NO_DETECTABLE_DIFF       # one look: max_checkpoints=1
    r = RULE.decide(*paired((3, 2, 1, 2)), SPEC,
                    DecisionContext("e", paired=True, checkpoint=1, max_checkpoints=4))
    assert r.outcome is Outcome.INSUFFICIENT and r.needed_n and r.needed_n > 20


def simulate(cells_prob, n, sims, seed, max_checkpoints=1):
    """cells_prob = P(both, only_b, only_a, neither) for each pair."""
    rng = random.Random(seed)
    outcomes = []
    for _ in range(sims):
        counts = [0, 0, 0, 0]
        for _ in range(n):
            counts[rng.choices(range(4), weights=cells_prob)[0]] += 1
        ctx = DecisionContext("s", paired=True, checkpoint=1, max_checkpoints=max_checkpoints)
        outcomes.append(RULE.decide(*paired(tuple(counts)), SPEC, ctx).outcome)
    return outcomes


def test_false_positive_rate_under_null():
    # equal marginals (p=0.5), 20% discordant split evenly
    for n in (8, 30, 100):
        outs = simulate((0.4, 0.1, 0.1, 0.4), n=n, sims=1500, seed=n)
        wrong = sum(o in (Outcome.A_BETTER, Outcome.B_BETTER) for o in outs) / len(outs)
        assert wrong <= 0.05, (n, wrong)


def test_power_for_a_20_point_difference():
    # p_A = 0.5, p_B = 0.7
    outs = simulate((0.45, 0.25, 0.05, 0.25), n=60, sims=800, seed=1)
    assert sum(o is Outcome.B_BETTER for o in outs) / len(outs) > 0.8
