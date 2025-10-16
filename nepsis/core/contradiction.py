"""Contradiction density computation."""

import numpy as np
from numpy.typing import NDArray

from nepsis.core.types import State, Hypothesis


def build_contradiction_matrix(hypotheses: list[Hypothesis]) -> NDArray[np.float32]:
    """Build contradiction matrix Ξ encoding mutual exclusivity.

    Ξ[i,j] = 1 if hypotheses i and j are mutually exclusive, 0 otherwise.

    Args:
        hypotheses: List of hypotheses

    Returns:
        Contradiction matrix (symmetric, zero diagonal)
    """
    n = len(hypotheses)
    xi = np.zeros((n, n), dtype=np.float32)

    # Default: assume some hypotheses are mutually exclusive
    # In a real implementation, this would be domain-specific
    # For now, mark as mutually exclusive if very different priors
    for i in range(n):
        for j in range(i + 1, n):
            # Simple heuristic: different priors suggest different processes
            prior_diff = abs(hypotheses[i].prior - hypotheses[j].prior)

            # Mark as contradictory if priors differ significantly
            if prior_diff > 0.3:
                xi[i, j] = 1.0
                xi[j, i] = 1.0

    return xi


def compute_contradiction_density(
    state: State,
    contradiction_matrix: NDArray[np.float32] | None = None
) -> float:
    """Compute contradiction density ρ.

    Implements: ρ = Σ Ξ[h1,h2] · (π_post[h1] · π_post[h2])

    Measures how much probability mass is assigned to mutually exclusive hypotheses.

    Args:
        state: Current reasoning state
        contradiction_matrix: Mutual exclusivity matrix (if None, builds default)

    Returns:
        Contradiction density in [0, 1]
    """
    if contradiction_matrix is None:
        contradiction_matrix = build_contradiction_matrix(state.hypotheses)

    # Compute outer product of posteriors
    posterior_product = np.outer(state.posteriors, state.posteriors)

    # Weight by contradiction matrix
    contradictions = contradiction_matrix * posterior_product

    # Sum all contradictions (divide by 2 since matrix is symmetric)
    rho = float(np.sum(contradictions) / 2.0)

    return np.clip(rho, 0.0, 1.0)


def identify_contradictions(
    state: State,
    threshold: float = 0.1,
    contradiction_matrix: NDArray[np.float32] | None = None
) -> list[tuple[str, str, float]]:
    """Identify specific hypothesis pairs that contradict.

    Args:
        state: Current reasoning state
        threshold: Minimum posterior product to report
        contradiction_matrix: Mutual exclusivity matrix

    Returns:
        List of (hypothesis1_id, hypothesis2_id, contradiction_strength)
    """
    if contradiction_matrix is None:
        contradiction_matrix = build_contradiction_matrix(state.hypotheses)

    contradictions = []

    n = len(state.hypotheses)
    for i in range(n):
        for j in range(i + 1, n):
            if contradiction_matrix[i, j] > 0:
                strength = state.posteriors[i] * state.posteriors[j]
                if strength >= threshold:
                    contradictions.append(
                        (state.hypotheses[i].id, state.hypotheses[j].id, float(strength))
                    )

    # Sort by strength
    contradictions.sort(key=lambda x: x[2], reverse=True)

    return contradictions
