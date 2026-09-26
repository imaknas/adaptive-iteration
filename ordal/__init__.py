"""ordal — Domain-agnostic adaptive experimentation framework.

Statistics come from scipy where it has them. Hypotheses come from a Proposer you pass in;
results are judged by a DecisionRule (default: WelchIntervalRule).

    from ordal import (
        Ledger, Experiment, Variant, MetricSpec, Observation,
        Evaluator, WelchIntervalRule, Outcome,
        HypothesisEngine, Proposal, VariableRegistry, VariableDef,
    )
"""
from .assignment import assign
from .decision import (
    Decision,
    DecisionContext,
    DecisionRule,
    Outcome,
    PairedProportionRule,
    ProportionIntervalRule,
    RuleResult,
    Sample,
    WelchIntervalRule,
)
from .evaluator import Evaluator
from .evidence import EvidenceSummary, VariableEvidence, build_evidence
from .experiment import Experiment, Variant
from .hypothesis import HypothesisEngine, Proposal, Proposer, ReviewedProposal
from .ledger import Ledger
from .lifecycle import abandon, restart
from .loop import Assignment, Loop, TickReport
from .metrics import MetricSpec, Observation
from .registry import DuplicateDetector, TokenSetDetector, VariableDef, VariableRegistry
from .screening import Screening, estimate_capacity

__version__ = "0.11.0"

__all__ = [
    "Decision", "DecisionContext", "DecisionRule", "Outcome", "PairedProportionRule",
    "ProportionIntervalRule", "RuleResult", "Sample", "WelchIntervalRule", "Evaluator",
    "EvidenceSummary", "VariableEvidence", "build_evidence", "Experiment", "Variant",
    "HypothesisEngine", "Proposal", "Proposer", "ReviewedProposal", "Ledger",
    "MetricSpec", "Observation", "DuplicateDetector", "TokenSetDetector", "VariableDef",
    "VariableRegistry", "assign", "Screening", "estimate_capacity", "Assignment", "Loop",
    "TickReport", "abandon", "restart",
]
