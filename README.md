# adaptive_iteration

**Domain-agnostic adaptive experimentation: experiment → measure → judge → propose.**

A small Python framework for running an endless loop of A/B experiments in any
domain — short videos, emails, proposals — without fooling yourself. It owns the
parts that should not depend on your domain or your tools:

- **Judging results honestly.** Winners are declared from per-unit data with a
  confidence interval, at fixed weekly checkpoints, with the false-positive rate
  controlled across repeated looks. Missing or immature data is excluded, never
  counted as zero. "No difference" and "don't know yet" are different outcomes.
- **Keeping evidence in one place.** An append-only JSONL ledger holds every
  observation and every decision, so any verdict can be recomputed later.
- **Keeping the vocabulary stable.** A variable registry stops the same idea
  from being tested three times under three names.

It deliberately does **not** decide where hypotheses come from. You inject a
`Proposer`: a language model, a parameter grid, a rules engine, or a person.
`core/` uses the standard library only.

---

## Install

```bash
uv add adaptive-iteration      # or: pip install adaptive-iteration
```

Python 3.10+. No runtime dependencies.

---

## The loop

```python
from datetime import timedelta
from pathlib import Path
from adaptive_iteration import (
    Evaluator, HypothesisEngine, Ledger, MetricSpec, VariableDef, VariableRegistry,
)

ledger = Ledger(Path("data/ledger.jsonl"))
spec = MetricSpec(name="avg_view_pct", min_effect=3.0)   # smallest difference that matters

# 1. Register the variables you know about (proposers may add more, see below)
VariableRegistry(ledger, "shorts").register(
    VariableDef("hook_style", "how the first line grabs attention",
                execution="script prompt: opening sentence template"))

# 2. Ask your proposer for candidates; the engine reviews them against the registry
engine = HypothesisEngine(ledger, proposer=my_proposer)
for r in engine.generate("shorts", spec, n=3):
    print(r.status, r.variable, r.flags, r.reason)
experiment = engine.accept(next(r for r in engine.generate("shorts", spec) if r.accepted))
ledger.start_experiment(experiment.id)

# 3. Produce units, then record what you measure — per unit, not averages
ledger.record_observations(my_adapter.collect_observations(experiment))

# 4. Judge (safe to call daily; it only decides at weekly checkpoints)
decision = Evaluator(ledger).evaluate(experiment.id, spec)
print(decision.outcome, decision.effect, decision.interval, decision.reason)
```

`examples/quickstart.py` runs the whole loop on simulated data with no model.

---

## Judging

`Evaluator` decides **when** and **which data**; a `DecisionRule` decides **what
the data says**.

| Setting | Default | Meaning |
|---|---|---|
| `window` | 7 days | judge once per window after the experiment starts |
| `maturity` | 72 hours | a unit counts only if observed this long after it was produced |
| `max_windows` | 4 | at the 4th checkpoint an undecided experiment closes |

The default rule, `WelchIntervalRule`, builds a confidence interval for the
difference B − A (Welch t for interleaved experiments, paired t for paired ones)
and compares it with the region of practical equivalence `±min_effect`:

| Outcome | When |
|---|---|
| `B_BETTER` / `A_BETTER` | the whole interval lies beyond `+min_effect` / `−min_effect` |
| `EQUIVALENT` | the whole interval lies inside `±min_effect` |
| `INSUFFICIENT` | anything else before the last checkpoint (with an estimate of how many more units are needed) |
| `NO_DETECTABLE_DIFF` | anything else at the last checkpoint |

Alpha is split across the checkpoints (Bonferroni), so looking every week keeps the
experiment-wide false-positive rate under 5%. Closing one experiment never stops the
loop — the next hypothesis is always generated.

To use a different rule (Bayesian, sequential, domain-specific), pass any object
with `name` and `decide(a, b, spec, ctx) -> RuleResult`:

```python
Evaluator(ledger, rule=MyBayesianRule())
```

---

## Proposers

```python
class Proposer(Protocol):
    def propose(self, evidence: EvidenceSummary, n: int) -> list[Proposal]: ...
```

`EvidenceSummary` is plain data: every variable's status (`untested`, `open`,
`concluded`, `equivalent`, `no_detectable_diff`, `legacy_unverified`), best variant,
effect and interval, plus the registry, metric dispersion and data-quality counts.
`evidence.to_markdown()` renders it for a prompt if you want one.

`examples/llm_proposer.py` shows a model-backed proposer that takes any
`(system, user) -> str` function, so the model, client and prompt stay yours.

### Review

`HypothesisEngine.generate()` reviews each proposal before you see it:

| Status | Meaning |
|---|---|
| `known` | uses a registered variable (aliases are normalised to the canonical name) |
| `merged` | proposed as new, but duplicates a registered variable — mapped onto it |
| `new` | a genuinely new variable; flagged `needs_execution` if it says nothing about how to run it |
| `rejected` | unregistered without a definition, or the variable already has an open experiment |

Nothing is written until `accept()`, so unused proposals never pollute the registry.
The default duplicate check compares name tokens; inject your own
`DuplicateDetector` for semantic matching, or merge by hand:

```python
VariableRegistry(ledger, "shorts").merge("intro_visual_style", into="opening_visual_style")
```

---

## Adapters

The only layer that talks to your systems:

```python
class DomainAdapter(Protocol):
    def collect_observations(self, experiment: Experiment) -> list[Observation]: ...
```

Report unavailable metrics as `None`, never `0`. See `adapters/short_video.py`.

---

## Migrating from 0.1

0.1 ledgers stored per-arm averages and a caller-supplied winner flag, which cannot
be re-judged. A 0.1 file opens read-only; convert it with:

```python
from adaptive_iteration.migrate import v1_to_v2
v1_to_v2(Path("data/adaptive_ledger.json"), Path("data/ledger.jsonl"))
```

Old results become `legacy_unverified` evidence and their variable names are
registered. `Analyzer`, `DomainAdapter.get_signals/format_context` and the built-in
OpenAI call are gone; write a proposer instead.

---

## Design

See `docs/design/v0.2.md`.
