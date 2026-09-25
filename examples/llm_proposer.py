"""Example: a Proposer backed by a language model.

This lives outside the package on purpose — the framework never calls a model.
Copy it, pick your own client/model/prompt, and inject it into HypothesisEngine.
`complete` is any function (system_prompt, user_prompt) -> str, so the same class
works with OpenAI, Anthropic, a local model, or a test stub.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from adaptive_iteration import EvidenceSummary, Proposal, VariableDef, Variant

SYSTEM = """\
You design the next A/B experiments. Each experiment tests exactly ONE variable.
Prefer registered variables. To introduce a new one, set "new_variable" with a
description explaining how it differs from every registered variable, and say how
the pipeline would execute it if you know. Skip variables whose status is "open".
Return ONLY a JSON array of objects:
{"variable": str, "description": str,
 "variant_a": {"label": str, "hint": str}, "variant_b": {"label": str, "hint": str},
 "tier": 1|2, "mode": "interleaved"|"paired", "rationale": str,
 "new_variable": null | {"description": str, "execution": str | null}}
"""


class LLMProposer:
    def __init__(self, complete: Callable[[str, str], str]) -> None:
        self.complete = complete

    def propose(self, evidence: EvidenceSummary, n: int) -> list[Proposal]:
        raw = self.complete(SYSTEM, f"{evidence.to_markdown()}\n\nPropose {n} experiments.")
        items = json.loads(re.sub(r"^```[a-z]*\n?|\n?```$", "", raw.strip()))
        proposals = []
        for it in items:
            nv = it.get("new_variable")
            proposals.append(Proposal(
                variable=it["variable"],
                description=it.get("description", ""),
                variant_a=Variant(it["variant_a"]["label"], hint=it["variant_a"].get("hint")),
                variant_b=Variant(it["variant_b"]["label"], hint=it["variant_b"].get("hint")),
                tier=int(it.get("tier", 2)),
                mode=it.get("mode", "interleaved"),
                rationale=it.get("rationale", ""),
                new_variable=VariableDef(it["variable"], nv.get("description", ""),
                                         execution=nv.get("execution")) if nv else None,
            ))
        return proposals


if __name__ == "__main__":
    # Wiring with the OpenAI SDK (pip install openai); any other client works the same way.
    from openai import OpenAI

    client = OpenAI()

    def complete(system: str, user: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content

    proposer = LLMProposer(complete)
    print("LLMProposer ready:", proposer)
