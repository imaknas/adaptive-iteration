# Tutorial: your first experiment

This guide assumes you know some Python and nothing about this framework or about
statistics. By the end you will have run an experiment, read its verdict, and
checked how far to trust it.

The running example is a newsletter. You want to know whether subject lines phrased
as a **question** get more clicks than plain **statements**. The complete code is in
[`examples/newsletter.py`](../examples/newsletter.py); run it with
`python examples/newsletter.py`.

---

## 1. Is this the right tool?

Use it when all of these are true:

- You keep producing **the same kind of thing** over and over: emails, posts,
  videos, proposals, model outputs. Each one is a **unit**.
- Each unit can be **measured with a number** within days or a few weeks.
- **You choose** which version each unit gets, and you don't hand-pick which units
  get which version (alternating or random is fine).
- You produce **roughly 10 or more units per version per week**. Fewer still works,
  but each experiment will take months.

Don't use it for website traffic splitting (tools like GrowthBook or Statsig do that
better), for one-off decisions, or for outcomes you only see months later.

---

## 2. Five words you need

| Word | Meaning | In the example |
|---|---|---|
| **unit** | one thing you produced and measured | one email sent to one reader |
| **variable** | the thing you change | `subject_style` |
| **variant** | one version of the variable; each experiment compares A with B | `statement` (A) vs `question` (B) |
| **metric** | the number that says which is better | `clicked`: 1 if the reader clicked, else 0 |
| **min_effect** | the smallest difference you would actually act on | 5 percentage points |

Every verdict is one of five **outcomes**:

| Outcome | What it means | What you do |
|---|---|---|
| `b_better` / `a_better` | confidently better, by a meaningful amount | adopt the winner |
| `equivalent` | confidently less different than `min_effect` | keep whichever is cheaper |
| `insufficient` | can't tell yet | keep collecting |
| `no_detectable_diff` | ran out of time without a clear answer | stop; the difference, if any, is small |

---

## 3. Install

```bash
uv add adaptive-iteration      # or: pip install adaptive-iteration
```

The only dependency is scipy. Nothing here calls a language model unless you make it.

---

## 4. Step by step

### Step 1: say what "better" means

```python
from adaptive_iteration import MetricSpec

SPEC = MetricSpec(name="clicked", min_effect=0.05)
```

`min_effect` is the most important number you will choose, and only you can choose
it. Ask: *"If B beat A by this much, would I switch?"* Too small and experiments
never finish; too large and you miss real improvements. For a 0/1 metric it is in
proportion units, so `0.05` means 5 percentage points.

### Step 2: create a ledger and name your variables

```python
from pathlib import Path
from adaptive_iteration import Ledger, VariableDef, VariableRegistry

ledger = Ledger(Path("data/ledger.jsonl"))
VariableRegistry(ledger, "newsletter").register(VariableDef(
    "subject_style", "how the subject line is phrased",
    execution="template picked in send_campaign.py"))
```

The **ledger** is one append-only file holding everything: experiments, every
measurement, every verdict. Nothing in it is ever edited, so any verdict can be
recomputed later.

The **registry** gives every variable one official name. Without it, the same idea
tested as `subject_style` today and `subject_phrasing` next month would split its
evidence in two.

### Step 3: define and start an experiment

```python
from adaptive_iteration import Experiment, Variant

exp = Experiment(domain="newsletter", variable="subject_style",
                 variant_a=Variant("statement"), variant_b=Variant("question"))
ledger.add_experiment(exp)
ledger.start_experiment(exp.id)
```

Change **only this variable** while the experiment runs. If you also switch email
provider or audience halfway through, the results mix both changes.

### Step 4: record one row per unit

Whatever sends your emails decides the variant (alternate, or pick at random), then
you record what happened:

```python
from adaptive_iteration import Observation

ledger.record_observation(Observation(
    experiment_id=exp.id,
    variant="question",
    unit_id="email-2026-10-05-0042",
    produced_at="2026-10-05T09:14:00+00:00",   # when it went out
    observed_at="2026-10-08T09:00:00+00:00",   # when you read the metric
    metrics={"clicked": 1.0},
))
```

Two rules matter:

- **One row per unit, not averages.** The framework needs the spread between units
  to know how much to trust a difference.
- **Missing is `None`, never `0`.** If you couldn't read a unit's metric, record
  `None`. A zero would be counted as a real, terrible result.

You can record the same unit again later (for example once a day). The newest
reading wins.

### Step 5: ask for a verdict

```python
from datetime import timedelta
from adaptive_iteration import Evaluator, ProportionIntervalRule

evaluator = Evaluator(ledger, rule=ProportionIntervalRule(),
                      window=timedelta(days=7), maturity=timedelta(days=2))
decision = evaluator.evaluate(exp.id, SPEC)
print(decision.outcome, decision.reason)
```

Run this as often as you like, say from a daily cron job. It only makes a decision
**once per window** (weekly here) and records it. In between it answers
`insufficient`. This matters: if you checked every day and stopped the first time
things looked good, you would crown false winners far more often than you think.

`maturity` is how long a unit needs before its metric settles. Emails get most of
their clicks within two days, so units younger than that are not counted yet.

The example prints:

```
week 1: insufficient  n=200/200  interval [-0.0302, 0.15] overlaps ±0.05 (p_a=0.12, p_b=0.18)
week 2: b_better      n=400/400  B better: effect +0.0825, interval [0.02, 0.145] above +0 (p_a=0.105, p_b=0.188)
```

How to read it:

- **effect**: B minus A. Questions clicked 8.25 points more.
- **interval**: the range the true difference plausibly lies in. In week 1 it ran
  from −3 to +15 points, which is compatible with "no difference", so there was no
  verdict yet. By week 2 the whole range is above zero and the estimate exceeds
  `min_effect`, so B wins.

### Step 6: decide what to test next

Once an experiment closes, pick the next variable. You can keep your own list, or
plug in anything that proposes ideas, including a language model. See **Proposers**
in the [README](../README.md) and
[`examples/llm_proposer.py`](../examples/llm_proposer.py). The framework checks
every proposal against the registry and refuses to reopen a variable that is
already being tested.

### Step 7: check the setup against your own data

Before trusting any verdict, ask two questions about your own numbers:

1. When nothing is different, how often would this setup crown a winner anyway?
2. When something really is better by `min_effect`, how often does it notice, and
   how fast?

`calibrate` answers both by replaying your history thousands of times:

```python
from adaptive_iteration.replay import calibrate

history = [1.0] * 12 + [0.0] * 88      # last quarter: 12% of emails were clicked
better  = [1.0] * 17 + [0.0] * 83      # the same, plus 5 points

calibrate(history, ProportionIntervalRule(), SPEC, effect=0, per_window=200)
calibrate(history, ProportionIntervalRule(), SPEC, effect=0.05, pool_b=better,
          per_window=200)
```

The example prints:

```
false winners when nothing differs: 1.0%
a real +5 points caught: 63%, median 3.0 weeks
```

If the second number is low, you need more units per week, a larger `min_effect`,
or more patience (`max_windows`).

---

## 5. Choosing the settings

| Setting | Where | How to choose |
|---|---|---|
| `min_effect` | `MetricSpec` | the smallest difference you would act on |
| rule | `Evaluator(rule=...)` | `ProportionIntervalRule()` for 0/1 metrics, `WelchIntervalRule()` (the default) for everything else |
| `window` | `Evaluator` | long enough for about 10+ units per variant; 7 days is a good start |
| `maturity` | `Evaluator` | how long until a unit's metric stops changing much |
| `max_windows` | `Evaluator` | how many windows before you give up; default 4 |

## 6. When your units come in groups

Sometimes units fall into groups that differ a lot on their own. For example, videos
about cooking might average 100% viewed while videos about history average 60%. If
one variant happens to get more cooking videos, it will look better for reasons that
have nothing to do with the variant.

Tag each unit with its group:

```python
Observation(..., metrics={"avg_view_pct": 97.0}, stratum="cooking")
```

The default `WelchIntervalRule` then compares A with B **within** each group and
combines the results, so an uneven mix cannot pose as an effect. Groups that only one
variant reached are left out and reported in the verdict. `ProportionIntervalRule`
does not use groups yet. For 0/1 metrics, assign variants so every group gets both
equally (for example, alternate within each group).

## 7. Common mistakes

- **Writing 0 for missing data.** Record `None`.
- **Changing something else mid-experiment.** Start a new experiment after any other
  change (new template, new model, new audience).
- **Stopping as soon as it looks good.** Let the Evaluator decide at its checkpoints.
- **Recording averages.** Record each unit.
- **Letting the mix of groups drift between variants.** Use `stratum` or balance
  the assignment.

## 8. FAQ

**It keeps saying `insufficient`.** The `reason` shows the interval and
`decision.needed_n` estimates how many more units per variant would likely settle it.
If that number is out of reach, raise `min_effect` or accept a
`no_detectable_diff`.

**Can I use my own statistics?** Yes. Pass any object with a `name` and a
`decide(a, b, spec, ctx)` method as `rule=`. Before switching, compare it with the
current one using `adaptive_iteration.replay.gate`.

**Does it need an API key?** No. Only if you choose to write a proposer that calls
a model.
