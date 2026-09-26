"""Judging inference experiments: speed, quantization quality, and picking a winner.

Simulated data, no model required:  python examples/inference_benchmark.py

Three questions that come up when tuning local inference, each answered with a rule
called directly on one finished batch (no Evaluator, no schedule):

1. Is config B faster?          Same prompts under both configs → paired.
                                Context length differs a lot → stratified.
2. Is the quantized model       Same eval items under both → paired 0/1. The claim
   "no worse"?                  is "no meaningful loss", so the answer we want is
                                EQUIVALENT, not merely "no significant difference".
3. Which of many kernel         Picking the best of many noisy results overstates
   strategies is best, and      its gain → report the winner's-curse-corrected
   by how much?                 effect next to the measured one.

Measure in alternating order (A, B, A, B …), not all A then all B: machine load and
temperature drift over a run, and alternating spreads that drift over both arms.
"""
from __future__ import annotations

import random

from adaptive_iteration import (
    DecisionContext,
    MetricSpec,
    PairedProportionRule,
    Sample,
    WelchIntervalRule,
)
from adaptive_iteration.core.shrinkage import approx_se, estimate_prior

ONE_LOOK = DecisionContext("batch", paired=True, checkpoint=1, max_checkpoints=1)
rng = random.Random(0)


def speed() -> None:
    """tokens/s for 60 prompts × 3 context lengths, config A vs B, paired by prompt."""
    spec = MetricSpec(name="tok_per_s", min_effect=1.0)          # < 1 tok/s isn't worth it
    base = {"2k": 42.0, "16k": 30.0, "64k": 14.0}               # speed falls with context
    a, b, pairs = [], [], []
    for ctx_len, mean in base.items():
        for i in range(60):
            prompt = rng.gauss(0, 2.0)                           # some prompts are just slower
            drift = rng.gauss(0, 1.5)                            # load / temperature noise
            a.append(mean + prompt + drift)
            b.append(mean * 1.06 + prompt + rng.gauss(0, 1.5))  # B is ~6% faster
            pairs.append(f"{ctx_len}-{i}")
    r = WelchIntervalRule().decide(Sample(tuple(a), pair_ids=tuple(pairs)),
                                   Sample(tuple(b), pair_ids=tuple(pairs)), spec, ONE_LOOK)
    print(f"1. speed:   {r.outcome.value:<20} {r.reason}")


def quantization_quality(n_items: int) -> None:
    """Eval items answered by the full-precision and the quantized model."""
    spec = MetricSpec(name="correct", min_effect=0.02)           # lose < 2 points = fine
    fp, q, ids = [], [], []
    for i in range(n_items):
        hard = rng.random() < 0.3                                # item difficulty, shared
        p = 0.45 if hard else 0.9
        right_fp = rng.random() < p
        # the quantized model mostly agrees with full precision, flipping ~3% of items
        right_q = right_fp if rng.random() > 0.03 else not right_fp
        fp.append(float(right_fp))
        q.append(float(right_q))
        ids.append(f"item{i}")
    r = PairedProportionRule().decide(Sample(tuple(fp), pair_ids=tuple(ids)),
                                      Sample(tuple(q), pair_ids=tuple(ids)), spec, ONE_LOOK)
    print(f"2. quality, {n_items} items: {r.outcome.value:<18} {r.reason}")
    if r.outcome.value == "equivalent":
        print("            → safe to say the quantized model loses less than 2 points")
    elif r.outcome.value == "no_detectable_diff":
        print("            → NOT evidence of no loss: the data can't tell; add eval items")


def best_of_many(repeats: int = 100) -> None:
    """28 kernel strategies, each timed 8 times against the baseline; report the best.

    A single run can mislead either way, so this repeats the whole search and
    averages what was reported for the chosen strategy against its true gain."""
    spec = MetricSpec(name="speedup_pct", min_effect=2.0)
    measured, corrected, true = [], [], []
    for _ in range(repeats):
        results, truths = [], []
        for _ in range(28):
            true_gain = max(0.0, rng.gauss(3.0, 4.0))           # most do little, a few help
            ids = tuple(str(i) for i in range(8))
            base = [rng.gauss(0, 8.0) for _ in ids]
            new = [x + true_gain + rng.gauss(0, 8.0) for x in base]
            results.append(WelchIntervalRule().decide(Sample(tuple(base), pair_ids=ids),
                                                      Sample(tuple(new), pair_ids=ids),
                                                      spec, ONE_LOOK))
            truths.append(true_gain)
        prior = estimate_prior(results)
        i = max(range(28), key=lambda k: results[k].effect)
        measured.append(results[i].effect)
        corrected.append(prior.shrink(results[i].effect, approx_se(results[i])))
        true.append(truths[i])
    avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(f"3. best of 28, averaged over {repeats} searches: measured +{avg(measured):.1f}%, "
          f"corrected +{avg(corrected):.1f}%, true +{avg(true):.1f}%")


if __name__ == "__main__":
    speed()
    quantization_quality(500)
    quantization_quality(3000)
    best_of_many()
