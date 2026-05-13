"""core/hypothesis.py — HypothesisEngine: ledger + analysis → new Experiment candidates.

Uses OpenAI Chat API. Prompt structure:
  system  : role + output schema
  user    : past experiment results + top/bottom performer context

Output: list of Experiment dicts (JSON), one per hypothesis candidate.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .analyzer import AnalysisResult
from .experiment import Experiment, Variant
from .ledger import Ledger

_SYSTEM_PROMPT = """\
You are an adaptive experimentation strategist. Your job is to analyse past A/B experiment \
results and suggest the next highest-value hypotheses to test.

Rules:
- Each hypothesis must test exactly ONE variable (to avoid confounding).
- Build on what worked: if a variable already has a known winner, either skip it or propose \
  a refined follow-up.
- Prioritise variables with high variance in the metrics — they have the most room for \
  improvement.
- Avoid re-testing already conclusive experiments unless the context has meaningfully changed.
- Output ONLY a valid JSON array. Each element must match:

{
  "variable":    "<what is being tested, snake_case>",
  "description": "<one-sentence hypothesis>",
  "variant_a":   {"label": "<control label>", "params": {}, "hint": "<optional>"},
  "variant_b":   {"label": "<challenger label>", "params": {}, "hint": "<optional>"},
  "tier":        <1 | 2 | 3>,
  "rationale":   "<why this is worth testing now>"
}

Tier meanings:
  1 = high-impact, must run alone
  2 = moderate-impact, can run in parallel with other Tier 2
  3 = no A/B needed, recommend directly based on research
"""


class HypothesisEngine:
    """Generate new Experiment candidates from past ledger data.

    Parameters
    ----------
    ledger      : Ledger instance (source of truth for past results)
    api_key     : OpenAI API key (or path to a file containing it)
    model       : OpenAI chat model to use
    max_candidates : how many hypotheses to request
    """

    def __init__(
        self,
        ledger: Ledger,
        api_key: str | Path,
        model: str = "gpt-5.4-mini",
        max_candidates: int = 5,
    ) -> None:
        self.ledger = ledger
        self.model = model
        self.max_candidates = max_candidates

        if isinstance(api_key, Path) or (isinstance(api_key, str) and "\n" not in api_key and len(api_key) < 200):
            p = Path(api_key)
            if p.exists():
                self._api_key = p.read_text().strip()
            else:
                self._api_key = api_key  # treat as literal key string
        else:
            self._api_key = api_key

    # ── Public ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        domain: str,
        analysis: AnalysisResult,
        domain_context: str = "",
        extra_instructions: str = "",
    ) -> list[Experiment]:
        """Call LLM and return a list of Experiment candidates.

        Parameters
        ----------
        domain          : domain identifier (e.g. "short_video")
        analysis        : output of Analyzer.analyze()
        domain_context  : adapter-provided text about top/bottom performers
        extra_instructions : any domain-specific guidance to append to the prompt
        """
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)

        user_message = self._build_user_message(
            domain=domain,
            analysis=analysis,
            domain_context=domain_context,
            extra_instructions=extra_instructions,
        )

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            max_completion_tokens=2000,
        )

        raw = resp.choices[0].message.content.strip()
        candidates = self._parse_response(raw)
        experiments = []
        for c in candidates:
            exp = Experiment(
                domain=domain,
                variable=c.get("variable", "unknown"),
                description=c.get("description", ""),
                variant_a=Variant.from_dict(c.get("variant_a", {"label": "control"})),
                variant_b=Variant.from_dict(c.get("variant_b", {"label": "variant"})),
                tier=int(c.get("tier", 2)),
            )
            experiments.append(exp)

        return experiments

    # ── Internal ───────────────────────────────────────────────────────────────

    def _build_user_message(
        self,
        domain: str,
        analysis: AnalysisResult,
        domain_context: str,
        extra_instructions: str,
    ) -> str:
        parts: list[str] = [f"Domain: {domain}"]

        # Past results summary
        records = self.ledger.query(domain=domain)
        if records:
            concluded = [r for r in records if r.get("winner") is not None]
            parts.append(f"\n## Past experiments ({len(concluded)} concluded, {len(records)} total records)")
            # Group by experiment_id
            by_exp: dict[str, list] = {}
            for r in concluded:
                eid = r.get("experiment_id", "?")
                by_exp.setdefault(eid, []).append(r)
            for eid, exp_records in list(by_exp.items())[-10:]:  # last 10 experiments
                winners = [r for r in exp_records if r.get("winner")]
                w_str = f"winner={winners[0]['variant']} ({winners[0]['metric_values']})" if winners else "inconclusive"
                var = exp_records[0].get("variable", "?")
                parts.append(f"  - [{eid}] variable={var} → {w_str}")
        else:
            parts.append("\n## Past experiments: none yet — this is the first cycle.")

        # Winning patterns
        if analysis.winning_patterns:
            parts.append("\n## Known winners by variable\n" +
                         "\n".join(f"  {k}: {v}" for k, v in analysis.winning_patterns.items()))

        # High-variance dimensions
        if analysis.dimension_stats:
            parts.append("\n## Metric variance (highest = most opportunity)")
            for ds in analysis.dimension_stats[:5]:
                parts.append(f"  {ds.name}: variance={ds.variance:.3f}, mean={ds.mean:.3f} (n={ds.count})")

        # Domain-provided context (top/bottom performers)
        if domain_context:
            parts.append(f"\n## Domain context (top vs bottom performers)\n{domain_context}")

        parts.append(
            f"\n## Task\nSuggest {self.max_candidates} next experiment candidates for domain '{domain}'."
        )
        if extra_instructions:
            parts.append(f"\nAdditional guidance:\n{extra_instructions}")

        return "\n".join(parts)

    @staticmethod
    def _parse_response(raw: str) -> list[dict[str, Any]]:
        # Strip markdown code fences if present
        raw = re.sub(r"^```[a-z]*\n?|\n?```$", "", raw.strip())
        return json.loads(raw)
