"""Winner's-curse correction: empirical-Bayes shrinkage of reported effects."""
import random
import statistics

import pytest

from adaptive_iteration import Experiment, Ledger, MetricSpec, Outcome, Variant, build_evidence
from adaptive_iteration.core.decision import Decision
from adaptive_iteration.core.shrinkage import MIN_EXPERIMENTS, approx_se, estimate_prior

Z95 = 1.959963984540054


def decision(effect, se, exp_id="e"):
    lo, hi = effect - Z95 * se, effect + Z95 * se
    outcome = (Outcome.B_BETTER if lo > 0 else Outcome.A_BETTER if hi < 0
               else Outcome.NO_DETECTABLE_DIFF)
    return Decision(exp_id, outcome, "m", 1, 50, 50, {}, "r", {}, "x",
                    "2026-09-01T00:00:00+00:00", effect=effect, interval=(lo, hi),
                    confidence=0.95)


def test_approx_se_recovers_the_standard_error():
    assert approx_se(decision(3.0, 2.5)) == pytest.approx(2.5)


def test_no_correction_without_enough_history():
    prior = estimate_prior([decision(5.0, 2.0)] * (MIN_EXPERIMENTS - 1))
    assert prior.tau is None and prior.shrink(5.0, 2.0) is None
    assert "no correction" in prior.basis


def test_pure_noise_history_shrinks_everything_to_zero():
    rng = random.Random(1)
    ds = [decision(rng.gauss(0, 3.0), 3.0) for _ in range(200)]   # no real effects at all
    prior = estimate_prior(ds)
    assert prior.tau < 1.0
    assert abs(prior.shrink(6.0, 3.0)) < 0.6        # a "big" measured effect was mostly noise


def test_precise_estimates_barely_move():
    prior = estimate_prior([decision(e, 1.0) for e in (-8, -4, 0, 4, 8, 6, -6)])
    assert prior.shrink(6.0, 0.2) == pytest.approx(6.0, rel=0.01)
    # the noisier the estimate, the further it is pulled toward 0
    assert 0 < prior.shrink(6.0, 5.0) < prior.shrink(6.0, 2.0) < prior.shrink(6.0, 0.2)
    t2 = prior.tau ** 2
    assert prior.shrink(6.0, 5.0) == pytest.approx(6.0 * t2 / (t2 + 25.0))


def test_corrects_the_winners_curse():
    """True effects ~ N(0, 4); each measured with se 3. Among declared winners the raw
    effect overstates the truth; the corrected one is close to unbiased and more accurate."""
    rng = random.Random(7)
    truth, ds = [], []
    for i in range(2000):
        t = rng.gauss(0, 4.0)
        truth.append(t)
        ds.append(decision(t + rng.gauss(0, 3.0), 3.0, f"e{i}"))
    prior = estimate_prior(ds)
    assert prior.tau == pytest.approx(4.0, rel=0.1)

    winners = [(d, t) for d, t in zip(ds, truth) if d.outcome is Outcome.B_BETTER]
    raw_bias = statistics.fmean(d.effect - t for d, t in winners)
    shrunk_bias = statistics.fmean(prior.shrink(d.effect, 3.0) - t for d, t in winners)
    assert raw_bias > 1.0                       # winners look better than they are
    assert abs(shrunk_bias) < raw_bias / 3      # correction removes most of that
    raw_err = statistics.fmean((d.effect - t) ** 2 for d, t in zip(ds, truth))
    shrunk_err = statistics.fmean((prior.shrink(d.effect, 3.0) - t) ** 2
                                  for d, t in zip(ds, truth))
    assert shrunk_err < raw_err


def test_evidence_reports_corrected_effects_without_changing_verdicts(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    rng = random.Random(3)
    for i in range(12):
        exp = Experiment(domain="d", variable=f"v{i}", variant_a=Variant("a"),
                         variant_b=Variant("b"))
        ledger.add_experiment(exp)
        ledger.start_experiment(exp.id, at="2026-09-01T00:00:00+00:00")
        d = decision(rng.gauss(0, 4.0) + rng.gauss(0, 3.0), 3.0, exp.id)
        ledger.record_decision(d)
    ev = build_evidence(ledger, "d", MetricSpec("m", min_effect=1.0))
    assert ev.shrinkage["tau"] is not None and ev.shrinkage["n_experiments"] == 12
    for v in ev.variables:
        final = ledger.final_decision(v.experiments[0])
        assert v.status in ("concluded", "no_detectable_diff")          # verdicts as recorded
        assert abs(v.shrunk_effect) <= abs(final.effect) + 1e-12       # only ever pulled in
    assert "corrected" in ev.to_markdown()
