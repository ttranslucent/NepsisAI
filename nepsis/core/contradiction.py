"""Contradiction density computation (vectorized).

Implements ρ = 0.5 * (p^T Ξ p) where:
- p is posterior probability vector
- Ξ is exclusivity matrix (mutual exclusivity between hypotheses)
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from nepsis.core.types import State
from nepsis.core.utils import ensure_index, ensure_exclusivity


def compute_contradiction_density(state: State) -> float:
    """Compute contradiction density ρ (vectorized).

    Implements: ρ = 0.5 * Σ_{i≠j} Ξ[i,j] * p[i] * p[j]
              = 0.5 * (p^T Ξ p)   [since diagonal is zero]

    Measures probability mass assigned to mutually exclusive hypotheses.

    Args:
        state: Current reasoning state

    Returns:
        Contradiction density in [0, 1]

    Example:
        >>> # High exclusivity + high posteriors → high ρ
        >>> state.exclusivity.matrix[0,1] = 0.9  # mutually exclusive
        >>> state.posteriors = np.array([0.6, 0.5])  # both likely
        >>> rho = compute_contradiction_density(state)
        >>> # ρ ≈ 0.5 * 0.9 * 0.6 * 0.5 = 0.135
    """
    if not state.hypotheses or len(state.hypotheses) < 2:
        state.contradiction_density = 0.0
        return 0.0

    # Ensure exclusivity matrix is initialized and synced
    ensure_index(state)
    ensure_exclusivity(state)

    # Get posterior array in correct order
    p = state.posteriors

    # Vectorized: ρ = 0.5 * (p^T Ξ p)
    Ξ = state.exclusivity.matrix
    rho = 0.5 * float(p @ Ξ @ p)

    # Clamp to [0, 1]
    rho = max(0.0, min(1.0, rho))

    # Update state
    state.contradiction_density = rho

    return rho


def identify_contradictions(
    state: State,
    threshold: float = 0.1,
) -> list[tuple[str, str, float]]:
    """Identify specific hypothesis pairs that contradict.

    Args:
        state: Current reasoning state
        threshold: Minimum posterior product to report

    Returns:
        List of (hypothesis1_id, hypothesis2_id, contradiction_strength)
        sorted by strength (descending)

    Example:
        >>> contradictions = identify_contradictions(state, threshold=0.05)
        >>> for h1, h2, strength in contradictions:
        ...     print(f"{h1} vs {h2}: {strength:.3f}")
    """
    ensure_index(state)
    ensure_exclusivity(state)

    contradictions = []
    Ξ = state.exclusivity.matrix
    p = state.posteriors
    ids = [h.id for h in state.hypotheses]

    n = len(state.hypotheses)
    for i in range(n):
        for j in range(i + 1, n):
            if Ξ[i, j] > 0:
                # Contradiction strength = exclusivity × posterior product
                strength = float(Ξ[i, j] * p[i] * p[j])
                if strength >= threshold:
                    contradictions.append((ids[i], ids[j], strength))

    # Sort by strength (descending)
    contradictions.sort(key=lambda x: x[2], reverse=True)

    return contradictions


# Deprecated: kept for backward compatibility
def build_contradiction_matrix(hypotheses: list) -> NDArray[np.float32]:
    """DEPRECATED: Use exclusivity_builder.infer_exclusivity_from_expectations() instead.

    This function used prior differences as a heuristic, which is statistically
    meaningless. Use explicit exclusivity rules or expectation-based inference.
    """
    import warnings
    warnings.warn(
        "build_contradiction_matrix() is deprecated. "
        "Use exclusivity_builder.build_exclusivity_from_rules() or "
        "infer_exclusivity_from_expectations() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    n = len(hypotheses)
    return np.zeros((n, n), dtype=np.float32)
