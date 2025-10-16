"""
NepsisAI: Medical-Grade Reasoning Architecture

A non-ergodic reasoning framework for decision-making under irreversible risk.

Status: Research prototype - NOT validated for clinical use.
"""

__version__ = "0.1.0"
__author__ = "Trenten Don Thorn"

# Core exports (will be implemented)
from nepsis.core.kernel import reason, step
from nepsis.core.types import State, Signal, Hypothesis

__all__ = [
    "reason",
    "step",
    "State",
    "Signal",
    "Hypothesis",
]
