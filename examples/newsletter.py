"""Tutorial example: which subject-line style gets more clicks? (docs/tutorial.md)

Simulated data, no model, runs in a second:  python examples/newsletter.py

Each email sent is one unit. The metric is "clicked" (0/1), so the example uses
ProportionIntervalRule. Readers come from two segments that click at very different
rates; because the variant simply alternates, both arms get the same segment mix.
"""
from __future__ import annotations

import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from adaptive_iteration import (
    Evaluator,
    Experiment,
    Ledger,
    MetricSpec,
    Observation,
    ProportionIntervalRule,
    VariableDef,
    VariableRegistry,
    Variant,
)
from adaptive_iteration.replay import calibrate

DOMAIN = "newsletter"

# 1. What counts as better, and by how much it must differ to matter.
SPEC = MetricSpec(name="clicked", min_effect=0.05)          # 5 percentage points

# Simulated truth: questions really do get +8 points of clicks.
CLICK_RATE = {("question", "new"): 0.10, ("question", "loyal"): 0.38,
              ("statement", "new"): 0.02, ("statement", "loyal"): 0.30}


def main() -> None:
    rng = random.Random(1)
    ledger = Ledger(Path(tempfile.mkdtemp()) / "ledger.jsonl")

    # 2. Name the things you vary, once.
    VariableRegistry(ledger, DOMAIN).register(VariableDef(
        "subject_style", "how the subject line is phrased",
        execution="template picked in send_campaign.py"))

    # 3. Define and start one experiment.
    exp = Experiment(domain=DOMAIN, variable="subject_style",
                     variant_a=Variant("statement"), variant_b=Variant("question"))
    ledger.add_experiment(exp)
    start = datetime(2026, 10, 5, tzinfo=timezone.utc)
    ledger.start_experiment(exp.id, at=start.isoformat())

    evaluator = Evaluator(ledger, rule=ProportionIntervalRule(),
                          window=timedelta(days=7), maturity=timedelta(days=2))

    # 4. Every week: send 400 emails, alternate the variant, record one row per email.
    for week in range(1, 5):
        for i in range(400):
            variant = "question" if i % 2 else "statement"
            segment = "loyal" if rng.random() < 0.3 else "new"
            sent = start + timedelta(days=7 * (week - 1), minutes=20 * i)
            clicked = rng.random() < CLICK_RATE[(variant, segment)]
            ledger.record_observation(Observation(
                experiment_id=exp.id, variant=variant, unit_id=f"w{week}-{i}",
                produced_at=sent.isoformat(),
                observed_at=(sent + timedelta(days=3)).isoformat(),
                metrics={"clicked": 1.0 if clicked else 0.0}))

        # 5. Ask for a verdict. Safe to call daily; it only decides once per window.
        d = evaluator.evaluate(exp.id, SPEC, now=start + timedelta(days=7 * week))
        print(f"week {week}: {d.outcome.value:<13} n={d.n_a}/{d.n_b}  {d.reason}")
        if d.outcome.is_final:
            break

    # 6. Before trusting the setup: with your own history, how often would this
    #    rule crown a winner when nothing differs, and how often catch a real +5?
    history = [1.0] * 12 + [0.0] * 88                 # e.g. last quarter: 12% clicked
    better = [1.0] * 17 + [0.0] * 83
    null = calibrate(history, ProportionIntervalRule(), SPEC, effect=0, per_window=200,
                     sims=300)
    real = calibrate(history, ProportionIntervalRule(), SPEC, effect=0.05, pool_b=better,
                     per_window=200, sims=300)
    print(f"\nfalse winners when nothing differs: {null.wrong_direction:.1%}")
    print(f"a real +5 points caught: {real.detected:.0%}, "
          f"median {real.median_weeks_to_final} weeks")


if __name__ == "__main__":
    main()
