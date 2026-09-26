"""core/hypothesis.py — HypothesisEngine: evidence → injected Proposer → reviewed proposals.

The framework does not know where hypotheses come from. A Proposer may call a
language model, enumerate a parameter grid, apply rules, or ask a human; it only
has to turn an EvidenceSummary into Proposals. The engine's job is the part that
must not depend on the proposer: normalising variable names against the registry,
catching duplicates, refusing to reopen a variable that is already being tested, and
— given a Screening — refusing experiments that could never reach a verdict in time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

from .evidence import EvidenceSummary, build_evidence
from .experiment import Experiment, Variant
from .ledger import Ledger
from .metrics import MetricSpec
from .registry import DuplicateDetector, TokenSetDetector, VariableDef, VariableRegistry
from .screening import Capacity, Screening, estimate_capacity


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
    expected_effect: Optional[float] = None      # B − A the proposer expects, in metric units
    proposed_by: Optional[str] = None            # defaults to the proposer's name


class Proposer(Protocol):
    """Anything with propose(). An optional `name` attribute identifies it in the
    track record (evidence.proposers); otherwise its class name is used."""
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
    screening: Optional[dict[str, Any]] = None   # detectability assessment, if screened

    @property
    def accepted(self) -> bool:
        return self.status != "rejected"


class HypothesisEngine:
    def __init__(self, ledger: Ledger, proposer: Proposer,
                 duplicates: Optional[DuplicateDetector] = None,
                 screening: Optional[Screening] = None) -> None:
        self.ledger = ledger
        self.proposer = proposer
        self.duplicates = duplicates or TokenSetDetector()
        self.screening = screening

    @property
    def proposer_name(self) -> str:
        return getattr(self.proposer, "name", None) or type(self.proposer).__name__

    def generate(self, domain: str, spec: MetricSpec, n: int = 5) -> list[ReviewedProposal]:
        """Ask the proposer for *n* candidates and review each one. Nothing is written."""
        evidence = build_evidence(self.ledger, domain, spec)
        proposals = self.proposer.propose(evidence, n)
        registry = VariableRegistry(self.ledger, domain)
        open_vars = {
            registry.resolve(e.variable) or e.variable
            for e in self.ledger.experiments(domain=domain)
            if self.ledger.is_open(e.id)
        }
        capacity = (estimate_capacity(self.ledger, domain, spec, self.screening)
                     if self.screening else None)
        taken_in_batch: set[str] = set()
        reviewed = []
        for p in proposals:
            r = self._review(p, domain, registry, open_vars | taken_in_batch, spec, capacity)
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
                         variant_b=p.variant_b, tier=p.tier, mode=p.mode,
                         proposed_by=p.proposed_by or self.proposer_name,
                         expected_effect=p.expected_effect)
        self.ledger.add_experiment(exp)
        return exp

    def _review(self, p: Proposal, domain: str, registry: VariableRegistry,
                busy: set[str], spec: MetricSpec,
                capacity: Optional[Capacity]) -> ReviewedProposal:
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

        assessment = None
        if capacity is not None and self.screening is not None:
            assessment = capacity.assess(p.expected_effect, spec, self.screening)
            verdict = assessment["verdict"]
            if verdict == "below_min_effect":
                return ReviewedProposal(
                    p, domain, "rejected", None, (),
                    f"expected effect {p.expected_effect:g} is below min_effect "
                    f"{spec.min_effect:g}: even if right, not worth acting on", assessment)
            if verdict == "undetectable":
                return ReviewedProposal(
                    p, domain, "rejected", None, (),
                    f"an effect of {p.expected_effect:g} needs ~{assessment['needed_per_arm']} "
                    f"units per variant (~{assessment['windows_needed']} windows at "
                    f"~{assessment['units_per_arm_per_window']:.3g}/variant/window); the limit "
                    f"is {self.screening.max_windows} windows", assessment)
            if verdict in ("slow", "no_expected_effect", "unknown"):
                flags.append({"slow": "slow", "no_expected_effect": "no_expected_effect",
                              "unknown": "detectability_unknown"}[verdict])
        return ReviewedProposal(p, domain, status, canonical, tuple(flags), reason, assessment)
