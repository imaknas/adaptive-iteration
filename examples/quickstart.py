"""End-to-end run on simulated data, no model required: python examples/quickstart.py"""
from __future__ import annotations

import random
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from adaptive_iteration import (
    Evaluator,
    EvidenceSummary,
    HypothesisEngine,
    Ledger,
    MetricSpec,
    Observation,
    Proposal,
    VariableDef,
    VariableRegistry,
    Variant,
)


class GridProposer:
    """A Proposer needs no model: this one walks a fixed list of untested ideas."""

    IDEAS = [
        ("hook_style", "question", "scenario"),
        ("cta_style", "soft", "direct"),
    ]

    def propose(self, evidence: EvidenceSummary, n: int) -> list[Proposal]:
        tried = {v.variable for v in evidence.variables if v.status != "untested"}
        return [
            Proposal(variable=var, description=f"{a} vs {b}", variant_a=Variant(a),
                     variant_b=Variant(b))
            for var, a, b in self.IDEAS if var not in tried
        ][:n]


def main() -> None:
    rng = random.Random(0)
    ledger = Ledger(Path(tempfile.mkdtemp()) / "ledger.jsonl")
    spec = MetricSpec(name="avg_view_pct", min_effect=3.0)
    registry = VariableRegistry(ledger, "demo")
    registry.register(VariableDef("hook_style", "how the first line grabs attention"))
    registry.register(VariableDef("cta_style", "closing call to action"))

    engine = HypothesisEngine(ledger, GridProposer())
    reviewed = engine.generate("demo", spec, n=1)
    exp = engine.accept(reviewed[0])
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    ledger.start_experiment(exp.id, at=start.isoformat())

    evaluator = Evaluator(ledger)
    for week in range(1, 5):
        for i in range(20):  # 20 units per arm per week
            produced = start + timedelta(days=7 * (week - 1), hours=i * 8)
            for label, mean in ((exp.variant_a.label, 60.0), (exp.variant_b.label, 72.0)):
                ledger.record_observation(Observation(
                    experiment_id=exp.id, variant=label, unit_id=f"{label}-{week}-{i}",
                    produced_at=produced.isoformat(),
                    observed_at=(produced + timedelta(days=4)).isoformat(),
                    metrics={"avg_view_pct": rng.gauss(mean, 12)}))
        d = evaluator.evaluate(exp.id, spec, now=start + timedelta(days=7 * week + 4))
        print(f"week {week}: {d.outcome.value:18s} n={d.n_a}/{d.n_b}  {d.reason}")
        if d.outcome.is_final:
            break

    print()
    print(engine.generate("demo", spec)[0].proposal.variable, "is proposed next")


if __name__ == "__main__":
    main()
