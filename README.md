# adaptive_iteration

**Domain-agnostic adaptive experimentation framework.**

A lightweight Python framework that abstracts the **experiment → measure → learn → challenge**
cycle into reusable components. Bring your own domain; the framework handles the rest.

---

## What It Is

```
adaptive_iteration/
├── core/               # pure Python stdlib, zero domain deps
│   ├── experiment.py   # Experiment / Variant dataclasses + ExperimentState
│   ├── ledger.py       # JSON append-only results ledger
│   ├── analyzer.py     # top/bottom performer detection, variance per dimension
│   ├── hypothesis.py   # HypothesisEngine: ledger + analysis → LLM → Experiment candidates
│   └── config.py       # AdaptiveConfig: JSON config + winner hints
└── adapters/
    ├── base.py         # DomainAdapter ABC (3 methods to implement)
    └── short_video.py  # Example adapter: YouTube + Instagram (simulated data)
```

---

## Installation

```bash
# with uv (recommended)
uv add adaptive-iteration

# with pip
pip install adaptive-iteration
```

Requires Python 3.10+. The only runtime dependency is `openai` (used only in
`HypothesisEngine`; the rest of `core/` is stdlib-only).

---

## Quick Start

```python
import os
from pathlib import Path
from adaptive_iteration.core.ledger import Ledger
from adaptive_iteration.core.analyzer import Analyzer
from adaptive_iteration.core.hypothesis import HypothesisEngine
from adaptive_iteration.core.config import AdaptiveConfig

# 1. Load / create config
cfg = AdaptiveConfig(Path("data/adaptive_config.json"))
cfg.update({"domain": "my_domain", "primary_metric": "conversion_rate"})

# 2. Open ledger
ledger = Ledger(Path("data/adaptive_ledger.json"))

# 3. Record experiment results
ledger.append(
    domain="my_domain",
    experiment_id="exp001",
    variable="cta_style",
    variant="soft",
    metric_values={"conversion_rate": 4.2, "bounce_rate": 31.0},
    winner=True,
)

# 4. Analyse
analyzer = Analyzer(ledger, primary_metric="conversion_rate")
analysis = analyzer.analyze(domain="my_domain")
print(f"Top performers: {[p['variant'] for p in analysis.top_performers]}")
print(f"Winning patterns: {analysis.winning_patterns}")

# 5. Generate next hypotheses (requires OPENAI_API_KEY)
engine = HypothesisEngine(
    ledger=ledger,
    api_key=os.environ["OPENAI_API_KEY"],
)
candidates = engine.generate(domain="my_domain", analysis=analysis)
for exp in candidates:
    print(f"  [{exp.tier}] {exp.variable}: {exp.description}")
```

---

## Integrating a New Domain

Subclass `DomainAdapter` and implement three methods:

```python
from adaptive_iteration.adapters.base import DomainAdapter

class MyDomainAdapter(DomainAdapter):

    def collect_metrics(self, item_ids: list[str]) -> list[dict]:
        """Pull raw metrics from your external system for each item_id."""
        results = []
        for item_id in item_ids:
            raw = my_api.get_metrics(item_id)
            results.append({"id": item_id, **raw})
        return results

    def get_signals(self, metrics: list[dict]) -> dict:
        """Normalise to framework signals."""
        primary = sum(m["conversion_rate"] for m in metrics) / len(metrics)
        return {
            "primary_metric": primary,
            "secondary_metrics": {
                "bounce_rate": sum(m["bounce_rate"] for m in metrics) / len(metrics),
            },
        }

    def format_context(self, top_items, bottom_items) -> str:
        lines = ["Top performers:"]
        for item in top_items:
            lines.append(f"  [{item['id']}] conversion_rate={item.get('conversion_rate')}")
        lines.append("Bottom performers:")
        for item in bottom_items:
            lines.append(f"  [{item['id']}] conversion_rate={item.get('conversion_rate')}")
        return "\n".join(lines)
```

See `adapters/short_video.py` for a complete reference implementation.

---

## Experiment Modes

| Mode | When to use | Description |
|------|-------------|-------------|
| `interleaved` | Tier 2 | Rotate variants across different items; maximises volume |
| `paired` | Tier 1 | Generate two variants for the same item; cleaner causal inference |

Tier 1 experiments (high-impact variables) should use `paired` mode to eliminate
confounding factors introduced by item-level differences.

---

## Import Verification

```bash
python3 -c "from adaptive_iteration.core.experiment import Experiment; print('ok')"
```

---

## Design Principles

1. `core/` has **zero external deps** — pure Python stdlib only (`openai` in `hypothesis.py`
   is lazy-imported and optional until you call `HypothesisEngine`).
2. `DomainAdapter` is the **only** layer that touches external systems.
3. `Ledger` is the **single source of truth** — append-only JSON, no database required.
4. `HypothesisEngine` uses **structured JSON output** prompting so parsing is deterministic.
