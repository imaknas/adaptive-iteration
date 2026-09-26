"""adaptive_iteration — Domain-agnostic adaptive experimentation framework.

Statistics come from scipy where it has them. Hypotheses come from a Proposer you inject;
results are judged by a DecisionRule (default: WelchIntervalRule).

    from adaptive_iteration import (
        Ledger, Experiment, Variant, MetricSpec, Observation,
        Evaluator, WelchIntervalRule, Outcome,
        HypothesisEngine, Proposal, VariableRegistry, VariableDef,
    )
"""
from .core.decision import (
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
from .core.evaluator import Evaluator
from .core.evidence import EvidenceSummary, VariableEvidence, build_evidence
from .core.experiment import Experiment, Variant
from .core.hypothesis import HypothesisEngine, Proposal, Proposer, ReviewedProposal
from .core.ledger import Ledger
from .core.metrics import MetricSpec, Observation
from .core.registry import DuplicateDetector, TokenSetDetector, VariableDef, VariableRegistry

__version__ = "0.6.0"

__all__ = [
    "Decision", "DecisionContext", "DecisionRule", "Outcome", "PairedProportionRule",
    "ProportionIntervalRule", "RuleResult", "Sample", "WelchIntervalRule", "Evaluator",
    "EvidenceSummary", "VariableEvidence", "build_evidence", "Experiment", "Variant",
    "HypothesisEngine", "Proposal", "Proposer", "ReviewedProposal", "Ledger",
    "MetricSpec", "Observation", "DuplicateDetector", "TokenSetDetector", "VariableDef",
    "VariableRegistry",
]
