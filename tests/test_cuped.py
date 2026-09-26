"""CUPED covariate adjustment in WelchIntervalRule."""
import math
import random
import statistics
from datetime import datetime, timedelta, timezone

import pytest

from ordal import (
    Evaluator,
    Experiment,
    Ledger,
    MetricSpec,
    Observation,
    Outcome,
    Sample,
    Variant,
    WelchIntervalRule,
    service,
)
from ordal.decision import DecisionContext
from ordal.replay import replay

SPEC = MetricSpec(name="m", min_effect=2.0)
ONE = DecisionContext("c", paired=False, checkpoint=1, max_checkpoints=1)
RULE = WelchIntervalRule()


def arm(rng, n, effect, rho, sd=10.0):
    """Values whose correlation with the covariate is rho (covariate ~ N(0, 1))."""
    xs = [rng.gauss(0, 1) for _ in range(n)]
    ys = [50 + effect + rho * sd * x + math.sqrt(1 - rho ** 2) * sd * rng.gauss(0, 1)
          for x in xs]
    return Sample(tuple(ys), covariates=tuple(xs))


def test_matches_statsmodels_regression_and_interval():
    sm = pytest.importorskip("statsmodels.api")
    ws = pytest.importorskip("statsmodels.stats.weightstats")
    rng = random.Random(0)
    a, b = arm(rng, 60, 0, 0.7), arm(rng, 70, 3, 0.7)
    r = RULE.decide(a, b, SPEC, ONE)
    xs = list(a.covariates) + list(b.covariates)
    ys = list(a.values) + list(b.values)
    slope = sm.OLS(ys, sm.add_constant(xs)).fit().params[1]
    assert r.params["cuped"]["theta"] == pytest.approx(slope, rel=1e-9)
    mx = statistics.fmean(xs)
    adj_a = [y - slope * (x - mx) for y, x in zip(a.values, a.covariates)]
    adj_b = [y - slope * (x - mx) for y, x in zip(b.values, b.covariates)]
    cm = ws.CompareMeans(ws.DescrStatsW(adj_b), ws.DescrStatsW(adj_a))
    assert r.interval == pytest.approx(cm.tconfint_diff(alpha=0.05, usevar="unequal"),
                                       abs=1e-9)


def test_noise_shrinks_by_about_rho_squared():
    rng = random.Random(1)
    a, b = arm(rng, 2000, 0, 0.7), arm(rng, 2000, 0, 0.7)
    adj = RULE.decide(a, b, SPEC, ONE)
    raw = WelchIntervalRule(adjust=False).decide(a, b, SPEC, ONE)
    assert adj.params["cuped"]["variance_reduction"] == pytest.approx(0.49, abs=0.05)
    width = lambda r: r.interval[1] - r.interval[0]  # noqa: E731
    assert width(adj) / width(raw) == pytest.approx(math.sqrt(1 - 0.49), abs=0.05)


def simulate(effect, adjust, sims, n=20, windows=4, seed=0):
    rng = random.Random(seed)
    rule = WelchIntervalRule(adjust=adjust)
    outs = []
    for _ in range(sims):
        ya, xa, yb, xb = [], [], [], []
        for k in range(1, windows + 1):
            sa, sb = arm(rng, n, 0, 0.7), arm(rng, n, effect, 0.7)
            ya += sa.values
            xa += sa.covariates
            yb += sb.values
            xb += sb.covariates
            ctx = DecisionContext("s", paired=False, checkpoint=k, max_checkpoints=windows)
            out = rule.decide(Sample(tuple(ya), covariates=tuple(xa)),
                              Sample(tuple(yb), covariates=tuple(xb)), SPEC, ctx).outcome
            if out.is_final:
                break
        outs.append(out)
    return outs


def test_false_positive_rate_stays_within_alpha():
    outs = simulate(effect=0.0, adjust=True, sims=1500, seed=2)
    wrong = sum(o in (Outcome.A_BETTER, Outcome.B_BETTER) for o in outs) / len(outs)
    assert wrong <= 0.05


def test_detects_more_real_effects_with_the_same_data():
    adj = simulate(effect=4.0, adjust=True, sims=500, seed=3)
    raw = simulate(effect=4.0, adjust=False, sims=500, seed=3)
    rate = lambda outs: sum(o is Outcome.B_BETTER for o in outs) / len(outs)  # noqa: E731
    assert rate(adj) > rate(raw) + 0.15


def test_interval_coverage():
    rng = random.Random(4)
    hits = 0
    for _ in range(1000):
        lo, hi = RULE.decide(arm(rng, 40, 0, 0.7), arm(rng, 40, 3.0, 0.7), SPEC, ONE).interval
        hits += lo <= 3.0 <= hi
    assert 0.93 <= hits / 1000 <= 0.97


def test_skips_when_covariates_are_missing_or_imbalanced():
    rng = random.Random(5)
    a, b = arm(rng, 30, 0, 0.7), arm(rng, 30, 0, 0.7)
    holey = Sample(a.values, covariates=(None,) + a.covariates[1:])
    r = RULE.decide(holey, b, SPEC, ONE)
    assert r.params["cuped"]["applied"] is False and "missing" in r.reason
    shifted = Sample(b.values, covariates=tuple(x + 5 for x in b.covariates))  # not random
    r = RULE.decide(a, shifted, SPEC, ONE)
    assert r.params["cuped"]["applied"] is False and "imbalanced" in r.reason
    plain = RULE.decide(Sample(a.values), Sample(b.values), SPEC, ONE)
    assert "cuped" not in plain.params          # no covariates, nothing to say


def test_paired_experiments_are_not_adjusted():
    rng = random.Random(6)
    a, b = arm(rng, 20, 0, 0.7), arm(rng, 20, 3, 0.7)
    ids = tuple(str(i) for i in range(20))
    ctx = DecisionContext("p", paired=True, checkpoint=1, max_checkpoints=1)
    r = RULE.decide(Sample(a.values, pair_ids=ids, covariates=a.covariates),
                    Sample(b.values, pair_ids=ids, covariates=b.covariates), SPEC, ctx)
    assert "cuped" not in r.params


def test_covariate_survives_ledger_evaluator_replay_and_service(tmp_path):
    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    ledger = Ledger(tmp_path / "l.jsonl")
    exp = Experiment(domain="d", variable="v", variant_a=Variant("a"), variant_b=Variant("b"),
                     started=t0.isoformat())
    ledger.add_experiment(exp)
    ledger.start_experiment(exp.id, at=t0.isoformat())
    rng = random.Random(7)
    obs = []
    for i in range(40):
        for label, eff in (("a", 0), ("b", 4)):
            s = arm(rng, 1, eff, 0.7)
            t = t0 + timedelta(hours=i)
            obs.append(Observation(exp.id, label, f"{label}{i}", t.isoformat(),
                                   (t + timedelta(days=4)).isoformat(), {"m": s.values[0]},
                                   covariate=s.covariates[0]))
    ledger.record_observations(obs)
    d = Evaluator(ledger).evaluate(exp.id, SPEC, now=t0 + timedelta(days=8))
    assert d.rule_params["cuped"]["applied"] is True
    [rd, *_] = replay(ledger.experiment(exp.id), obs, SPEC)
    assert rd.rule_params["cuped"]["applied"] is True           # replay keeps covariates

    L = str(tmp_path / "s.jsonl")
    service.configure(L, "d", "m", 2.0)
    service.register_variable(L, "d", "v")
    eid = service.accept_proposal(L, "d", {"variable": "v", "variant_a": "a",
                                           "variant_b": "b"})["experiment"]["id"]
    row = {"experiment_id": eid, "variant": "a", "unit_id": "u", "produced_at": t0.isoformat(),
           "metrics": {"m": 1.0}}
    assert service.record_observations(L, [{**row, "covariate": 0.3}]) == {"recorded": 1}
    with pytest.raises(service.ServiceError, match="covariate"):
        service.record_observations(L, [{**row, "unit_id": "w", "covariate": "high"}])
