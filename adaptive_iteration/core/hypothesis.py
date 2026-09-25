"""core/hypothesis.py — HypothesisEngine: evidence → injected Proposer → reviewed proposals.

The framework does not know where hypotheses come from. A Proposer may call a
language model, enumerate a parameter grid, apply rules, or ask a human; it only
has to turn an EvidenceSummary into Proposals. The engine's job is the part that
must not depend on the proposer: normalising variable names against the registry,
catching duplicates, and refusing to reopen a variable that is already being tested.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Protocol

from .evidence import EvidenceSummary, build_evidence
from .experiment import Experiment, Variant
from .ledger import Ledger
from .metrics import MetricSpec
from .registry import DuplicateDetector, TokenSetDetector, VariableDef, VariableRegistry


@dataclass(frozen=True)
class Proposal:
    variable: str
    description: str
    variant_a: Variant
    variant_b: Variant
    tier: int = 2
    mode: Literal["interleaved", "paired"] = "interleaved"
    new_variable: Optional[VariableDef] = None   # required when proposing an unregistered variable
    rationale: str = ""


class Proposer(Protocol):
    def propose(self, evidence: EvidenceSummary, n: int) -> list[Proposal]: ...


ReviewStatus = Literal["known", "merged", "new", "rejected"]


@dataclass(frozen=True)
class ReviewedProposal:
    proposal: Proposal
    domain: str
    status: ReviewStatus
    variable: Optional[str]                  # canonical name after review (None if rejected)
    flags: tuple[str, ...] = field(default_factory=tuple)
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.status != "rejected"


class HypothesisEngine:
    def __init__(self, ledger: Ledger, proposer: Proposer,
                 duplicates: Optional[DuplicateDetector] = None) -> None:
        self.ledger = ledger
        self.proposer = proposer
        self.duplicates = duplicates or TokenSetDetector()

    def generate(self, domain: str, spec: MetricSpec, n: int = 5) -> list[ReviewedProposal]:
        """Ask the proposer for *n* candidates and review each one. Nothing is written."""
        evidence = build_evidence(self.ledger, domain, spec)
        proposals = self.proposer.propose(evidence, n)
        registry = VariableRegistry(self.ledger, domain)
        open_vars = {
            registry.resolve(e.variable) or e.variable
            for e in self.ledger.experiments(domain=domain)
            if self.ledger.final_decision(e.id) is None
        }
        taken_in_batch: set[str] = set()
        reviewed = []
        for p in proposals:
            r = self._review(p, domain, registry, open_vars | taken_in_batch)
            if r.accepted and r.variable:
                taken_in_batch.add(r.variable)
            reviewed.append(r)
        return reviewed

    def accept(self, reviewed: ReviewedProposal) -> Experiment:
        """Register a new variable if needed and add the experiment to the ledger."""
        if not reviewed.accepted or reviewed.variable is None:
            raise ValueError(f"cannot accept a rejected proposal: {reviewed.reason}")
        p = reviewed.proposal
        if reviewed.status == "new":
            assert p.new_variable is not None
            VariableRegistry(self.ledger, reviewed.domain).register(p.new_variable)
        exp = Experiment(domain=reviewed.domain, variable=reviewed.variable,
                         description=p.description, variant_a=p.variant_a,
                         variant_b=p.variant_b, tier=p.tier, mode=p.mode)
        self.ledger.add_experiment(exp)
        return exp

    def _review(self, p: Proposal, domain: str, registry: VariableRegistry,
                busy: set[str]) -> ReviewedProposal:
        def rejected(reason: str) -> ReviewedProposal:
            return ReviewedProposal(p, domain, "rejected", None, (), reason)

        if p.variant_a.label == p.variant_b.label:
            return rejected("variant_a and variant_b have the same label")

        canonical = registry.resolve(p.variable)
        status: ReviewStatus
        flags: list[str] = []
        if canonical is not None:
            status, reason = "known", ""
            if canonical != p.variable:
                reason = f"{p.variable!r} is an alias of {canonical!r}"
        elif p.new_variable is None:
            return rejected(f"{p.variable!r} is not registered and no new_variable "
                            "definition was given")
        else:
            candidate = p.new_variable
            if candidate.name != p.variable:
                return rejected("new_variable.name must equal proposal.variable")
            match = self.duplicates.find_match(candidate, registry)
            if match is not None:
                canonical, status = match, "merged"
                reason = f"proposed {p.variable!r} duplicates registered {match!r}"
            else:
                canonical, status, reason = p.variable, "new", "new variable"
                if not candidate.execution:
                    flags.append("needs_execution")

        if canonical in busy:
            return rejected(f"{canonical!r} already has an open experiment")
        return ReviewedProposal(p, domain, status, canonical, tuple(flags), reason)
