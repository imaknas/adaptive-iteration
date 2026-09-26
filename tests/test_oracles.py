"""Every interval the rules produce, checked against an independent source.

- where a library implements the same method: statsmodels as the oracle
- where none does (paired Newcombe, post-stratified difference): coverage — the
  95% interval must contain the true effect about 95% of the time in simulation
"""
import random

import pytest

from adaptive_iteration import (
    DecisionContext,
    MetricSpec,
    PairedProportionRule,
    ProportionIntervalRule,
    Sample,
    WelchIntervalRule,
)

sm_prop = pytest.importorskip("statsmodels.stats.proportion")
sm_ws = pytest.importorskip("statsmodels.stats.weightstats")

SPEC = MetricSpec(name="m", min_effect=0.05)
ONE_LOOK = DecisionContext("oracle", paired=False, checkpoint=1, max_checkpoints=1)
ONE_LOOK_PAIRED = DecisionContext("oracle", paired=True, checkpoint=1, max_checkpoints=1)


def bern(k, n):
    return Sample(tuple([1.0] * k + [0.0] * (n - k)))


# ── Library oracles ───────────────────────────────────────────────────────────

def test_proportion_rule_matches_statsmodels_newcombe():
    rng = random.Random(0)
    for _ in range(200):
        n_a, n_b = rng.randint(5, 300), rng.randint(5, 300)
        k_a, k_b = rng.randint(0, n_a), rng.randint(0, n_b)
        r = ProportionIntervalRule().decide(bern(k_a, n_a), bern(k_b, n_b), SPEC, ONE_LOOK)
        lo, hi = sm_prop.confint_proportions_2indep(k_b, n_b, k_a, n_a, method="newcomb",
                                                    compare="diff", alpha=0.05)
        assert r.interval == pytest.approx((lo, hi), abs=1e-9), (k_a, n_a, k_b, n_b)


def test_welch_rule_matches_statsmodels():
    rng = random.Random(1)
    for _ in range(100):
        a = [rng.gauss(50, rng.uniform(1, 30)) for _ in range(rng.randint(5, 80))]
        b = [rng.gauss(55, rng.uniform(1, 30)) for _ in range(rng.randint(5, 80))]
        r = WelchIntervalRule().decide(Sample(tuple(a)), Sample(tuple(b)), SPEC, ONE_LOOK)
        cm = sm_ws.CompareMeans(sm_ws.DescrStatsW(b), sm_ws.DescrStatsW(a))
        assert r.interval == pytest.approx(cm.tconfint_diff(alpha=0.05, usevar="unequal"),
                                           abs=1e-9)


def test_paired_welch_matches_statsmodels():
    rng = random.Random(2)
    for _ in range(100):
        n = rng.randint(5, 60)
        base = [rng.gauss(50, 20) for _ in range(n)]
        a = base
        b = [x + 3 + rng.gauss(0, 5) for x in base]
        ids = tuple(str(i) for i in range(n))
        r = WelchIntervalRule().decide(Sample(tuple(a), pair_ids=ids),
                                       Sample(tuple(b), pair_ids=ids), SPEC, ONE_LOOK_PAIRED)
        diffs = [y - x for x, y in zip(a, b)]
        assert r.interval == pytest.approx(sm_ws.DescrStatsW(diffs).tconfint_mean(0.05),
                                           abs=1e-9)


# ── Coverage where no library exists ──────────────────────────────────────────

def coverage(draw, truth, sims):
    hits = 0
    for _ in range(sims):
        lo, hi = draw()
        hits += lo <= truth <= hi
    return hits / sims


@pytest.mark.parametrize("n", [30, 100])
@pytest.mark.parametrize("cells", [(0.45, 0.25, 0.05, 0.25), (0.40, 0.10, 0.10, 0.40),
                                   (0.80, 0.08, 0.02, 0.10)])
def test_paired_proportion_interval_coverage(n, cells):
    rng = random.Random(hash((n, cells)) & 0xFFFF)
    truth = cells[1] - cells[2]               # p_B − p_A = only_b − only_a

    def draw():
        a, b = [], []
        for _ in range(n):
            c = rng.choices(range(4), weights=cells)[0]
            a.append(1.0 if c in (0, 2) else 0.0)
            b.append(1.0 if c in (0, 1) else 0.0)
        ids = tuple(str(i) for i in range(n))
        return PairedProportionRule(min_n=2).decide(
            Sample(tuple(a), pair_ids=ids), Sample(tuple(b), pair_ids=ids),
            SPEC, ONE_LOOK_PAIRED).interval

    cov = coverage(draw, truth, sims=1500)
    assert 0.93 <= cov <= 0.99, (n, cells, cov)


def test_stratified_interval_coverage_under_confounded_allocation():
    rng = random.Random(3)
    truth = 4.0

    def draw():
        vals_a, st_a, vals_b, st_b = [], [], [], []
        for _ in range(120):
            s = "money" if rng.random() < 0.3 else "other"       # arm A: mostly "other"
            vals_a.append(rng.gauss(100 if s == "money" else 60, 12))
            st_a.append(s)
            s = "money" if rng.random() < 0.7 else "other"       # arm B: mostly "money"
            vals_b.append(rng.gauss((100 if s == "money" else 60) + truth, 12))
            st_b.append(s)
        return WelchIntervalRule().decide(Sample(tuple(vals_a), strata=tuple(st_a)),
                                          Sample(tuple(vals_b), strata=tuple(st_b)),
                                          MetricSpec("m", min_effect=3.0), ONE_LOOK).interval

    cov = coverage(draw, truth, sims=1000)
    assert 0.93 <= cov <= 0.97, cov
