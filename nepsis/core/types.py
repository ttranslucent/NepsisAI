"""Core data types for NepsisAI reasoning system."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from enum import Enum
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from nepsis.core.exclusivity_builder import Exclusivity


class SignalType(Enum):
    """Type of signal input."""
    VITAL = "vital"
    LAB = "lab"
    SYMPTOM = "symptom"
    IMAGING = "imaging"
    HISTORY = "history"
    RED = "red"  # Safety-critical


@dataclass
class Signal:
    """An observable signal/evidence."""
    type: SignalType | str
    name: str
    value: float
    timestamp: Optional[float] = None
    red_threshold: Optional[float] = None  # If set, triggers red channel
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Convert string type to enum if needed."""
        if isinstance(self.type, str):
            try:
                self.type = SignalType(self.type)
            except ValueError:
                pass  # Allow custom types


@dataclass
class Hypothesis:
    """A hypothesis about the world state."""
    id: str
    name: str
    prior: float = 0.0
    expects: Dict[str, Any] = field(default_factory=dict)  # Expected signal values
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate hypothesis."""
        if not 0.0 <= self.prior <= 1.0:
            raise ValueError(f"Prior must be in [0,1], got {self.prior}")


@dataclass
class Evidence:
    """Processed evidence for a hypothesis."""
    hypothesis_id: str
    likelihood: float
    coherence: float
    posterior: float
    interpretant_state: NDArray[np.float32]


class CollapseMode(Enum):
    """Decision collapse modes."""
    OCCAM = "occam"  # Single best hypothesis
    HICKAM = "hickam"  # Multiple hypotheses
    ZEROBACK = "zeroback"  # Epistemic reset


@dataclass
class State:
    """Complete reasoning state at a time step."""

    # Hypotheses
    hypotheses: List[Hypothesis]
    priors: NDArray[np.float32]
    posteriors: NDArray[np.float32]

    # Interpretant layer
    interpretant_states: NDArray[np.float32]
    interpretant_dim: int = 16

    # Evidence tracking
    likelihoods: NDArray[np.float32] = field(default_factory=lambda: np.array([]))
    coherence_scores: NDArray[np.float32] = field(default_factory=lambda: np.array([]))

    # Exclusivity matrix (mutual exclusivity between hypotheses)
    exclusivity: Optional[Exclusivity] = None

    # Internal index for fast hypothesis lookup
    _index: Dict[str, int] = field(default_factory=dict, repr=False)

    # Convergence metrics
    contradiction_density: float = 0.0
    lyapunov_value: float = 0.0
    lyapunov_stable: bool = False
    ruin_prob: float = 0.0  # Non-ergodic ruin probability (monotone)

    # Control
    step_num: int = 0
    collapse_mode: CollapseMode = CollapseMode.OCCAM
    red_preempted: bool = False

    # Metadata for collapse decisions and audit
    metadata: Dict[str, Any] = field(default_factory=dict)

    # History
    posterior_history: List[NDArray[np.float32]] = field(default_factory=list)

    @classmethod
    def from_hypotheses(
        cls,
        hypotheses: List[Hypothesis],
        interpretant_dim: int = 16
    ) -> "State":
        """Initialize state from hypothesis list."""
        n_hyp = len(hypotheses)

        # Extract priors
        priors = np.array([h.prior for h in hypotheses], dtype=np.float32)

        # Normalize if needed
        if priors.sum() > 0:
            priors = priors / priors.sum()
        else:
            priors = np.ones(n_hyp, dtype=np.float32) / n_hyp

        # Initialize posteriors to priors
        posteriors = priors.copy()

        # Initialize interpretant states (uniform)
        interpretant_states = np.ones(interpretant_dim, dtype=np.float32) / interpretant_dim

        # Initialize evidence
        likelihoods = np.ones(n_hyp, dtype=np.float32)
        coherence_scores = np.ones(n_hyp, dtype=np.float32)

        return cls(
            hypotheses=hypotheses,
            priors=priors,
            posteriors=posteriors,
            interpretant_states=interpretant_states,
            interpretant_dim=interpretant_dim,
            likelihoods=likelihoods,
            coherence_scores=coherence_scores,
        )

    def get_top_hypothesis(self, n: int = 1) -> List[Hypothesis]:
        """Get top n hypotheses by posterior probability."""
        indices = np.argsort(self.posteriors)[::-1][:n]
        return [self.hypotheses[i] for i in indices]

    def get_posterior(self, hypothesis_id: str) -> float:
        """Get posterior probability for a hypothesis."""
        for i, h in enumerate(self.hypotheses):
            if h.id == hypothesis_id:
                return float(self.posteriors[i])
        raise ValueError(f"Hypothesis {hypothesis_id} not found")


@dataclass
class ReasoningResult:
    """Final output of reasoning process."""
    state: State
    top_hypothesis: Hypothesis
    top_posterior: float
    contradiction_density: float
    converged: bool
    red_preempted: bool
    steps: int
    audit_trail: Optional[List[Dict[str, Any]]] = None
