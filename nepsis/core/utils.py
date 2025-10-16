"""Core utility functions for state management and indexing."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from nepsis.core.types import State

from nepsis.core.exclusivity_builder import Exclusivity, infer_exclusivity_from_expectations


def ensure_index(state: "State") -> None:
    """Ensure hypothesis index is synced with current hypotheses.

    Maintains O(1) lookup from hypothesis ID to array index.
    Rebuilds if hypotheses have changed.

    Args:
        state: State to update
    """
    hyp_ids = {h.id for h in state.hypotheses}
    if not state._index or set(state._index.keys()) != hyp_ids:
        state._index = {h.id: i for i, h in enumerate(state.hypotheses)}


def ensure_exclusivity(state: "State") -> None:
    """Ensure exclusivity matrix exists and is synced with hypotheses.

    If no exclusivity matrix is set, infers one from hypothesis expectations.
    Rebuilds if hypothesis set has changed.

    Args:
        state: State to update
    """
    ensure_index(state)

    # Check if we need to rebuild
    hyp_ids = {h.id for h in state.hypotheses}
    needs_rebuild = (
        state.exclusivity is None or
        set(state.exclusivity.ids) != hyp_ids
    )

    if needs_rebuild:
        ids = [h.id for h in state.hypotheses]

        if state.exclusivity is None:
            # No exclusivity set - infer from expectations
            state.exclusivity = infer_exclusivity_from_expectations(
                hypos={h.id: h for h in state.hypotheses}
            )
        else:
            # Exclusivity exists but hypotheses changed - preserve known pairs
            old_excl = state.exclusivity
            n = len(ids)
            M = np.zeros((n, n), dtype=np.float32)
            idx = {h: i for i, h in enumerate(ids)}

            # Copy known exclusivities
            for i, h1 in enumerate(ids):
                for j, h2 in enumerate(ids[i+1:], start=i+1):
                    if h1 in old_excl._index and h2 in old_excl._index:
                        M[i, j] = M[j, i] = old_excl.get(h1, h2)

            state.exclusivity = Exclusivity(ids=ids, matrix=M, default=old_excl.default)


def get_posterior_array(state: "State") -> np.ndarray:
    """Get posterior probabilities as ordered array.

    Args:
        state: State with posteriors

    Returns:
        Array of posteriors ordered by state._index
    """
    ensure_index(state)
    ids = [h.id for h in state.hypotheses.values()]
    return np.array([state.posteriors[i] for i in range(len(ids))], dtype=np.float32)


def update_posteriors_from_array(state: "State", posteriors: np.ndarray) -> None:
    """Update state posteriors from ordered array.

    Args:
        state: State to update
        posteriors: Array of posterior probabilities
    """
    ensure_index(state)
    if len(posteriors) != len(state.hypotheses):
        raise ValueError(f"Posterior array length {len(posteriors)} != hypotheses {len(state.hypotheses)}")

    state.posteriors = posteriors.astype(np.float32)
