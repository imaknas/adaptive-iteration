# Using ordal from an agent

An agent that improves something by experimenting (writing posts, emails, prompts,
code) needs a judge it can't argue with. Left to itself, a loop tends to declare
winners from noise, count missing data as zero, and quietly relax its own
standards. ordal gives the agent the same operations a person would use,
with those failure modes blocked.

Two ways in, with identical behaviour:

- **MCP server**: the agent calls tools directly.
- **CLI**: JSON in, JSON out, for scripts, cron jobs and agents that run shell
  commands.

Both work on one ledger file, the single record of every experiment, observation
and verdict.

---

## Connect over MCP

```bash
pip install "ordal[mcp]"          # or use uvx as below, no install
```

**Claude Code:**

```bash
claude mcp add ordal -- \
  uvx --from "ordal[mcp]" ordal \
  --ledger /absolute/path/to/ledger.jsonl mcp
```

**Any MCP client** (`mcpServers` config):

```json
{
  "mcpServers": {
    "ordal": {
      "command": "uvx",
      "args": ["--from", "ordal[mcp]", "ordal",
               "--ledger", "/absolute/path/to/ledger.jsonl", "mcp"]
    }
  }
}
```

The server sends the agent its workflow and rules as MCP instructions when it
connects, so no extra prompting is needed.

### Tools

| Tool | Writes? | Purpose |
|---|---|---|
| `configure` | yes | metric, `min_effect`, rule and schedule for a domain (`rule="proportion"` covers paired 0/1 experiments too) |
| `register_variable` | yes | name something you will vary |
| `merge_variables` | yes | declare two names the same variable |
| `review_proposals` | no | `{"reviewed": [...]}`: ideas checked against the registry and running experiments |
| `accept_proposal` | yes | start an experiment (returns its id); screened for detectability when it has an `expected_effect`; `start=false` leaves it waiting |
| `start_experiment` | yes | start one that is waiting (e.g. after the user approved it) |
| `record_observations` | yes | one row per produced unit |
| `evaluate` | at checkpoints | `{"results": [...]}`: verdict plus `next_action` per experiment, always judged at the current time |
| `assign_variant` | yes | which variant a unit gets; balanced within its stratum and recorded |
| `pending` | no | `{"pending": [...]}`: closed verdicts not yet put into effect |
| `mark_applied` | yes | record that a verdict is now in effect in the pipeline |
| `restart_experiment` | yes | the pipeline changed mid-experiment: abandon it and start it again from now (`start=false` to wait for approval) |
| `abandon_experiment` | yes | close an experiment with no verdict (never judged or applied) |
| `evidence` | no | what is known per variable, what it cost, and each proposer's track record |
| `status` | no | experiments per domain, and the settings **new** experiments will get; a running experiment keeps the settings it started with (shown in its `evaluate` output) |
| `calibrate` | no | false-winner rate and detection rate on the domain's own data (`seed` makes it repeatable) |

The CLI has the same operations plus two that agents don't get: `evaluate --now`
(judging as of another time, for shadow runs and scripts; over MCP an agent could use
it to jump to a checkpoint on demand) and `migrate` (a one-time file conversion).

---

## Use from the command line

```bash
export ORDAL_LEDGER=data/ledger.jsonl

ordal configure --domain newsletter --metric clicked \
    --min-effect 0.05 --rule proportion
ordal register-variable --domain newsletter --name subject_style \
    --execution "template picked in send_campaign.py"
ordal accept --domain newsletter \
    --json '{"variable": "subject_style", "variant_a": "statement", "variant_b": "question"}'

# one JSON object per unit, as a list or JSONL, from a file or stdin
cat sent_today.jsonl | ordal record

ordal evaluate --domain newsletter     # safe to run daily from cron
ordal evidence --domain newsletter --markdown
ordal calibrate --domain newsletter --effect 0 --per-window 200

ordal assign --experiment 3f9a1c2e --unit email-0043 --stratum loyal
ordal pending --domain newsletter
ordal mark-applied --experiment 3f9a1c2e

ordal accept --domain newsletter --no-start --json '{...}'   # wait for approval
ordal start --experiment 3f9a1c2e                            # approved: start it
ordal migrate data/adaptive_ledger.json data/ledger.jsonl    # old v0.1 file
```

Errors print `{"error": "..."}`, exit with status 1, and say how to fix the call.

An observation row:

```json
{"experiment_id": "3f9a1c2e", "variant": "question", "unit_id": "email-0042",
 "produced_at": "2026-10-05T09:14:00+00:00",
 "metrics": {"clicked": 1}, "stratum": "loyal"}
```

`observed_at` defaults to now. Record the same unit again later to update it.
Optional `"covariate"`: a number known before the variant was assigned that predicts
the metric; it makes verdicts arrive sooner. Never use anything measured after
assignment.

---

## Guardrails built in

These hold no matter what the agent asks for:

| Failure mode | What stops it |
|---|---|
| Declaring a winner from noise | verdicts only at fixed checkpoints, with a confidence interval; alpha is split across checkpoints |
| Acting on interim numbers | `evaluate` returns `closed: false` and a `next_action` that says not to act |
| Moving the goalposts | `configure` never changes a running experiment; it keeps the settings it started with |
| Missing data counted as 0 | `null` is required for missing values and is excluded, never zeroed |
| Half-written batches | `record_observations` validates every row first and writes all or nothing |
| Re-testing the same idea under a new name | proposals are checked against the variable registry; duplicates are merged or rejected |
| Two experiments on one variable at once | rejected at `accept_proposal` |
| Trusting an untested setup | `calibrate` measures false winners and detection on the domain's own data |
| Picking variants by hand | `assign_variant` decides; an observation contradicting a unit's assignment is refused |
| Starting experiments that can't finish | proposals with an `expected_effect` too small for the domain's volume are rejected |
| Judging across a pipeline change | `restart_experiment` starts over; data from before the restart never counts |

## What the agent is still responsible for

The framework can't see your pipeline, so the agent must:

- ask `assign_variant` for every unit's variant and use exactly that;
- give every proposal an honest `expected_effect` and a `proposed_by` name;
- change nothing else about the pipeline while an experiment runs; if something
  else does change, call `restart_experiment` and tell the user;
- record every unit, including the ones that did badly;
- ask the user for `min_effect` rather than guessing it.

These rules are also in the MCP instructions the agent receives on connect.
