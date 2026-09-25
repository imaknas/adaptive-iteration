"""adaptive_iteration — Domain-agnostic adaptive experimentation framework.

core/ is standard library only. Hypotheses come from a Proposer you inject;
results are judged by a DecisionRule (default: WelchIntervalRule).

    from adaptive_iteration import (
        Ledger, Experiment, Variant, MetricSpec, Observation,
        Evaluator, WelchIntervalRule, Outcome,
        HypothesisEngine, Proposal, VariableRegistry, VariableDef,
    )
"""
from .core.decision import (
    Decision,
    DecisionRule,
    Outcome,
    ProportionIntervalRule,
    WelchIntervalRule,
)
from .core.evaluator import Evaluator
from .core.evidence import EvidenceSummary, VariableEvidence, build_evidence
from .core.experiment import Experiment, Variant
from .core.hypothesis import HypothesisEngine, Proposal, Proposer, ReviewedProposal
from .core.ledger import Ledger
from .core.metrics import MetricSpec, Observation
from .core.registry import DuplicateDetector, TokenSetDetector, VariableDef, VariableRegistry

__version__ = "0.4.1"

__all__ = [
    "Decision", "DecisionRule", "Outcome", "ProportionIntervalRule", "WelchIntervalRule", "Evaluator",
    "EvidenceSummary", "VariableEvidence", "build_evidence", "Experiment", "Variant",
    "HypothesisEngine", "Proposal", "Proposer", "ReviewedProposal", "Ledger",
    "MetricSpec", "Observation", "DuplicateDetector", "TokenSetDetector", "VariableDef",
    "VariableRegistry",
]
