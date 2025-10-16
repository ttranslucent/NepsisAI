"""Lyapunov convergence tracking."""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from nepsis.core.types import State
from nepsis.utils.math import entropy


@dataclass
class LyapunovWeights:
    """Weights for Lyapunov function components."""
    contradiction: float = 2.0
    entropy: float = 1.0
    coherence: float = 1.5
    velocity: float = 0.5


def compute_lyapunov(
    state: State,
    weights: LyapunovWeights | None = None
) -> float:
    """Compute Lyapunov energy function.

    V = α_c·ρ + α_e·H(π) + α_h·|coh-π| + α_v·||Δπ||

    Reasoning converges when V decreases monotonically.

    Args:
        state: Current reasoning state
        weights: Component weights

    Returns:
        Lyapunov value (lower = more stable)
    """
    if weights is None:
        weights = LyapunovWeights()

    # Component 1: Contradiction density
    V_contradiction = weights.contradiction * state.contradiction_density

    # Component 2: Entropy (uncertainty)
    V_entropy = weights.entropy * entropy(state.posteriors)

    # Component 3: Coherence-posterior mismatch
    if len(state.coherence_scores) > 0:
        coherence_mismatch = np.mean(np.abs(state.coherence_scores - state.posteriors))
    else:
        coherence_mismatch = 0.0
    V_coherence = weights.coherence * coherence_mismatch

    # Component 4: Velocity (how fast beliefs are changing)
    if len(state.posterior_history) > 0:
        delta = state.posteriors - state.posterior_history[-1]
        velocity = np.linalg.norm(delta)
    else:
        velocity = 0.0
    V_velocity = weights.velocity * velocity

    # Total Lyapunov value
    V = V_contradiction + V_entropy + V_coherence + V_velocity

    return float(V)


def check_convergence(
    state: State,
    window: int = 3,
    tolerance: float = 0.01
) -> bool:
    """Check if reasoning has converged.

    Convergence requires:
    1. Lyapunov value decreasing
    2. Low contradiction density
    3. Stable posteriors

    Args:
        state: Current reasoning state
        window: Number of steps to check for stability
        tolerance: Maximum change allowed for convergence

    Returns:
        True if converged
    """
    # Need enough history
    if len(state.posterior_history) < window:
        return False

    # Check 1: Low contradiction
    if state.contradiction_density > 0.3:
        return False

    # Check 2: Stable posteriors (low velocity)
    recent_posteriors = state.posterior_history[-window:]
    changes = [
        np.linalg.norm(recent_posteriors[i] - recent_posteriors[i-1])
        for i in range(1, len(recent_posteriors))
    ]

    if max(changes) > tolerance:
        return False

    # Check 3: High confidence in top hypothesis
    if state.posteriors.max() < 0.5:
        return False

    return True


def lyapunov_gradient(
    state: State,
    prev_lyapunov: float,
    current_lyapunov: float
) -> float:
    """Compute Lyapunov gradient dV/dt.

    Negative gradient indicates convergence.

    Args:
        state: Current state
        prev_lyapunov: Previous Lyapunov value
        current_lyapunov: Current Lyapunov value

    Returns:
        Lyapunov gradient (negative = converging)
    """
    return current_lyapunov - prev_lyapunov
