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

**New here? Start with the [tutorial](docs/tutorial.md)**: a complete first
experiment, explained for someone who has never used the framework.

**Building an agent?** The same operations are available as an **MCP server** and a
**JSON CLI**, with guardrails an automated loop can't talk its way around (no acting
on interim numbers, no moving the goalposts mid-experiment). See
[docs/agents.md](docs/agents.md).

It deliberately does **not** decide where hypotheses come from. You inject a
`Proposer`: a language model, a parameter grid, a rules engine, or a person.
Standard statistics (t distribution, Welch intervals) come from scipy; the few
methods no library provides are implemented here and checked against published
examples and simulated coverage.

---

## Install

```bash
uv add adaptive-iteration      # or: pip install adaptive-iteration
```

Python 3.10+. Depends on scipy. For the MCP server:
`pip install "adaptive-iteration[mcp]"`.

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

## Running it automatically

`Loop` drives the whole cycle so nobody has to. You plug in three things and call
`tick()` on a schedule:

```python
from adaptive_iteration import Loop

loop = Loop(ledger, "shorts",
            collect=my_adapter.collect_observations,   # experiment -> observations
            apply=put_into_pipeline,                   # (experiment, variant, decision)
            proposer=my_proposer,
            max_concurrent=1,
            require_start_approval=True,               # new experiments wait for a human
            require_approval=True)                     # so do B-wins before they apply

# while producing each unit: which variant of each running experiment it gets
for a in loop.variant_for(unit_id="video-0412", stratum="cooking"):
    render_with(a.variable, a.variant)

report = loop.tick()          # from cron, daily is fine
```

Each tick collects fresh observations, judges every running experiment (still only
at checkpoints), puts closed verdicts into effect (the winner, or variant A when
nothing won), and fills free slots with new proposals. Before a proposal is started
it is **screened**: from the proposer's `expected_effect`, the domain's own spread
and its weekly volume, the loop estimates how long a verdict would take, and turns
away proposals that could never be detected in time or aren't worth acting on even
if right.

With `require_start_approval`, accepted proposals hold their slot until
`loop.approve_start(id)` (or `loop.reject_start(id)`); nothing is assigned or
collected before that.

If the pipeline changes under a running experiment (new model, prompt or config),
data from before and after can't be compared. `loop.restart(id, reason)` abandons
the experiment and starts the same one again, judged only on data from after the
change and under the settings in effect from then on. `loop.abandon(id, reason)`
drops one with no verdict. Neither is ever applied.

A copy of the ledger file plus `apply` as a no-op and a proposer that returns `[]`
runs the loop in **shadow**: it judges and reports what it would apply, and changes
nothing.

Nobody can know in advance whether a hypothesis is right, but a proposer's record
shows over time. `evidence()` keeps one per proposer: how its experiments ended, what
they cost, and how its expected effects compared with what was measured.

A simulated pipeline (60 units a week, noisy, topic-skewed) run for 16 weeks with no
human involved, 200 times: a real +15 improvement was adopted every time; changes
with no real effect were adopted 1.8% of the time.

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
| `B_BETTER` / `A_BETTER` | the interval excludes 0 **and** the estimated effect is at least `min_effect` |
| `EQUIVALENT` | the whole interval lies inside `±min_effect` |
| `INSUFFICIENT` | anything else before the last checkpoint (with an estimate of how many more units are needed) |
| `NO_DETECTABLE_DIFF` | anything else at the last checkpoint |

`WelchIntervalRule(superiority="margin")` is a stricter variant that requires the
whole interval to clear `min_effect`. Replayed on real production data (see replay
below), both kept false positives under 5%, but the stricter variant caught real
effects far less often, so it is not the default.

Alpha is split across the checkpoints (Bonferroni), so looking every week keeps the
experiment-wide false-positive rate under 5%. Closing one experiment never stops the
loop — the next hypothesis is always generated.

**0/1 metrics** (replied, clicked, converted) should use `ProportionIntervalRule`,
which builds Newcombe's interval for the difference in proportions. It stays honest
when events are rare: zero successes in both arms gives a wide interval, not a
false "equivalent".

```python
Evaluator(ledger, rule=ProportionIntervalRule())
```

**Paired 0/1 outcomes** (the same item scored under A and under B: a fact recalled
or not after two policies, one email in two versions) use `PairedProportionRule`,
Newcombe's paired score interval. Units are matched by `pair_id`, and a small sample
where A and B always agree reads as "not enough evidence", never as "equivalent".
`ProportionIntervalRule` switches to it automatically for paired experiments.

**Groups of units** that differ a lot on their own (topics, audience segments) can
be tagged with `Observation(..., stratum="cooking")`. `WelchIntervalRule` then
compares the arms within each group and combines the results, so an uneven mix of
groups between the arms cannot pose as an effect.

**Judging a finished batch in one look.** For offline experiments (every unit
already scored, no weekly schedule), call a rule directly and skip the Evaluator:

```python
from adaptive_iteration import DecisionContext, MetricSpec, PairedProportionRule, Sample

ids = ("fact1", "fact2", ...)                    # same order in both arms
a = Sample(values=(1.0, 0.0, ...), pair_ids=ids)  # policy A: recalled?
b = Sample(values=(1.0, 1.0, ...), pair_ids=ids)  # policy B
r = PairedProportionRule().decide(
    a, b, MetricSpec(name="recalled", min_effect=0.10),
    DecisionContext("batch-1", paired=True, checkpoint=1, max_checkpoints=1))
print(r.outcome, r.effect, r.interval, r.reason)
```

`max_checkpoints=1` means the full alpha is spent on this single look. If you will
look again after adding more data, set it to the total number of looks you plan.

To use a different rule (Bayesian, sequential, domain-specific), pass any object
with `name` and `decide(a, b, spec, ctx) -> RuleResult`:

```python
Evaluator(ledger, rule=MyBayesianRule())
```

---

## Judging the judge: replay

Before trusting a rule — or switching to a new one — test it on your own data.
Adapted from the replay idea in [Dream-RSI](https://arxiv.org/abs/2609.14858):
recorded history becomes a simulator, and a candidate is adopted only if it is not
worse than the incumbent.

```python
from adaptive_iteration.replay import calibrate, gate, replay

pool = [...]  # every real per-unit value of the metric you have

# How often does a rule crown a winner when nothing differs? How often does it
# catch a real 10-point effect, and after how many weeks?
calibrate(pool, WelchIntervalRule(), spec, effect=0,  per_window=20)
calibrate(pool, WelchIntervalRule(), spec, effect=10, per_window=20)

# Adopt only if false positives stay ≤ 5% and detection is not worse
gate(candidate_rule, current_rule, pool, spec, effects=(5, 10, 20), per_window=20)

# 0/1 metrics can't be shifted: give B its own pool instead
calibrate(clicks, ProportionIntervalRule(), spec, effect=0.05, pool_b=clicks_plus_5,
          per_window=200)

# What would the Evaluator have said, week by week, on a recorded experiment?
replay(experiment, observations, spec, rule=candidate_rule)
```

Replay only reuses outcomes that were actually observed. It can evaluate *how you
judge and schedule* experiments; it cannot predict how an untested hypothesis would
have done.

---

## Proposers

```python
class Proposer(Protocol):
    def propose(self, evidence: EvidenceSummary, n: int) -> list[Proposal]: ...
```

`EvidenceSummary` is plain data: every variable's status (`untested`, `open`,
`concluded`, `equivalent`, `no_detectable_diff`, `legacy_unverified`), best variant,
effect and interval, units still needed for open experiments, plus the registry,
metric dispersion, data-quality counts, and the cost of judging so far (units spent
per decisive result) — so a proposer can prefer hypotheses that resolve quickly.
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
