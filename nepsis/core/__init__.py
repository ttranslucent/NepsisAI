"""Core reasoning components for NepsisAI."""

from nepsis.core.kernel import reason, step
from nepsis.core.types import State, Signal, Hypothesis, Evidence
from nepsis.core.interpretant import apply_interpretant, coherence_score
from nepsis.core.contradiction import compute_contradiction_density
from nepsis.core.exclusivity_builder import (
    Exclusivity,
    build_exclusivity_from_rules,
    infer_exclusivity_from_expectations,
    exclusivity_from_pairdict,
)

__all__ = [
    "reason",
    "step",
    "State",
    "Signal",
    "Hypothesis",
    "Evidence",
    "apply_interpretant",
    "coherence_score",
    "compute_contradiction_density",
    "Exclusivity",
    "build_exclusivity_from_rules",
    "infer_exclusivity_from_expectations",
    "exclusivity_from_pairdict",
]
