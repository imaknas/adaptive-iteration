# Using adaptive-iteration from an agent

An agent that improves something by experimenting (writing posts, emails, prompts,
code) needs a judge it can't argue with. Left to itself, a loop tends to declare
winners from noise, count missing data as zero, and quietly relax its own
standards. adaptive-iteration gives the agent the same operations a person would use,
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
pip install "adaptive-iteration[mcp]"          # or use uvx as below, no install
```

**Claude Code:**

```bash
claude mcp add adaptive-iteration -- \
  uvx --from "adaptive-iteration[mcp]" adaptive-iteration \
  --ledger /absolute/path/to/ledger.jsonl mcp
```

**Any MCP client** (`mcpServers` config):

```json
{
  "mcpServers": {
    "adaptive-iteration": {
      "command": "uvx",
      "args": ["--from", "adaptive-iteration[mcp]", "adaptive-iteration",
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
| `accept_proposal` | yes | start an experiment (returns its id) |
| `record_observations` | yes | one row per produced unit |
| `evaluate` | at checkpoints | `{"results": [...]}`: verdict plus `next_action` per experiment |
| `evidence` | no | what is known per variable, and what it cost |
| `status` | no | experiments per domain |
| `calibrate` | no | false-winner rate and detection rate on the domain's own data |

---

## Use from the command line

```bash
export ADAPTIVE_ITERATION_LEDGER=data/ledger.jsonl

adaptive-iteration configure --domain newsletter --metric clicked \
    --min-effect 0.05 --rule proportion
adaptive-iteration register-variable --domain newsletter --name subject_style \
    --execution "template picked in send_campaign.py"
adaptive-iteration accept --domain newsletter \
    --json '{"variable": "subject_style", "variant_a": "statement", "variant_b": "question"}'

# one JSON object per unit, as a list or JSONL, from a file or stdin
cat sent_today.jsonl | adaptive-iteration record

adaptive-iteration evaluate --domain newsletter     # safe to run daily from cron
adaptive-iteration evidence --domain newsletter --markdown
adaptive-iteration calibrate --domain newsletter --effect 0 --per-window 200
```

Errors print `{"error": "..."}`, exit with status 1, and say how to fix the call.

An observation row:

```json
{"experiment_id": "3f9a1c2e", "variant": "question", "unit_id": "email-0042",
 "produced_at": "2026-10-05T09:14:00+00:00",
 "metrics": {"clicked": 1}, "stratum": "loyal"}
```

`observed_at` defaults to now. Record the same unit again later to update it.

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

## What the agent is still responsible for

The framework can't see your pipeline, so the agent must:

- assign variants by alternation or at random, never by picking units that
  "suit" a variant;
- change nothing else about the pipeline while an experiment runs, and tell the
  user if something else does change;
- record every unit, including the ones that did badly;
- ask the user for `min_effect` rather than guessing it.

These rules are also in the MCP instructions the agent receives on connect.
