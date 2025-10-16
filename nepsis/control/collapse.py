"""Collapse governor for decision-making with Hickam cluster support."""

from __future__ import annotations
import numpy as np
from typing import Tuple, List, Optional

from nepsis.core.types import State, CollapseMode, Hypothesis
from nepsis.core.config import (
    COLLAPSE_MIN_TOP,
    COLLAPSE_MAX_CONTRA,
    COLLAPSE_MAX_RUIN,
    HICKAM_MASS_THRESH,
    HICKAM_MAX_PAIR_XI,
)
from nepsis.core.utils import ensure_index, ensure_exclusivity


def _top(state: State) -> Tuple[Optional[Hypothesis], float]:
    """Get top hypothesis and its posterior.

    Returns:
        Tuple of (top_hypothesis, top_posterior) or (None, 0.0)
    """
    if not state.hypotheses or len(state.posteriors) == 0:
        return (None, 0.0)

    top_idx = np.argmax(state.posteriors)
    return (state.hypotheses[top_idx], float(state.posteriors[top_idx]))


def _hickam_cluster_ok(state: State) -> Tuple[bool, List[Hypothesis], float]:
    """Check if Hickam cluster collapse is valid.

    Greedy algorithm:
    1. Sort hypotheses by posterior (descending)
    2. Add each to cluster if exclusivity with all current members <= threshold
    3. Stop when total mass >= threshold and cluster size >= 2

    Args:
        state: Current reasoning state

    Returns:
        Tuple of (is_valid, cluster_hypotheses, total_mass)
    """
    ensure_index(state)
    ensure_exclusivity(state)

    if not state.hypotheses or len(state.hypotheses) < 2:
        return (False, [], 0.0)

    # Sort hypotheses by posterior (descending)
    sorted_indices = np.argsort(state.posteriors)[::-1]
    sorted_hyps = [state.hypotheses[i] for i in sorted_indices]

    cluster: List[Hypothesis] = []
    mass = 0.0
    Ξ = state.exclusivity.matrix

    for h in sorted_hyps:
        if state.posteriors[state._index[h.id]] == 0.0:
            break

        # Check pairwise exclusivity with current cluster members
        compatible = True
        for c in cluster:
            xi = Ξ[state._index[h.id], state._index[c.id]]
            if xi > HICKAM_MAX_PAIR_XI:
                compatible = False
                break

        if compatible:
            cluster.append(h)
            mass += state.posteriors[state._index[h.id]]

        # Success criterion: enough mass and multiple causes
        if mass >= HICKAM_MASS_THRESH and len(cluster) >= 2:
            return (True, cluster, mass)

    # Didn't meet criteria
    return (False, cluster, mass)


def decide_collapse(
    state: State,
    occam_threshold: Optional[float] = None,
    hickam_threshold: Optional[float] = None,
    zeroback_threshold: Optional[float] = None
) -> Tuple[CollapseMode, List[Hypothesis]]:
    """Decide which collapse mode to use (legacy compatibility).

    Args:
        state: Current reasoning state
        occam_threshold: Min posterior for Occam collapse (unused, uses config)
        hickam_threshold: Min posterior for Hickam inclusion (unused, uses config)
        zeroback_threshold: Max contradiction for avoiding ZeroBack (unused, uses config)

    Returns:
        Tuple of (collapse_mode, selected_hypotheses)
    """
    # Check for ZeroBack condition
    if state.contradiction_density > (zeroback_threshold or COLLAPSE_MAX_CONTRA):
        return (CollapseMode.ZEROBACK, [])

    # Check collapse based on mode
    if state.collapse_mode == CollapseMode.HICKAM:
        ok, cluster, mass = _hickam_cluster_ok(state)
        if ok:
            return (CollapseMode.HICKAM, cluster)

    # Occam or fallback
    top_hyp, top_w = _top(state)
    if top_hyp and top_w >= (occam_threshold or COLLAPSE_MIN_TOP):
        return (CollapseMode.OCCAM, [top_hyp])

    # No collapse
    return (state.collapse_mode, [])


def should_collapse(state: State) -> bool:
    """Decide if state should collapse to decision.

    Args:
        state: Current reasoning state

    Returns:
        True if collapse criteria met
    """
    top_hyp, top_w = _top(state)

    if top_hyp is None:
        return False

    # Never collapse in defer mode
    if state.collapse_mode == CollapseMode.ZEROBACK:
        # ZeroBack is handled separately
        return False

    # Occam collapse: single strong hypothesis, low contradiction, low ruin
    if state.collapse_mode == CollapseMode.OCCAM:
        return (
            top_w >= COLLAPSE_MIN_TOP and
            state.contradiction_density <= COLLAPSE_MAX_CONTRA and
            state.ruin_prob <= COLLAPSE_MAX_RUIN
        )

    # Hickam collapse: valid cluster with low ruin
    if state.collapse_mode == CollapseMode.HICKAM:
        ok, cluster, mass = _hickam_cluster_ok(state)
        return ok and state.ruin_prob <= COLLAPSE_MAX_RUIN

    return False


def collapse(state: State) -> State:
    """Apply collapse decision to state.

    Updates state.decision and logs collapse details.

    Args:
        state: State to collapse

    Returns:
        Updated state
    """
    if state.collapse_mode == CollapseMode.HICKAM:
        ok, cluster, mass = _hickam_cluster_ok(state)
        if ok:
            # Log Hickam cluster collapse
            cluster_ids = [h.id for h in cluster]
            state.metadata = state.metadata if hasattr(state, 'metadata') else {}
            state.metadata['collapse_cluster'] = cluster_ids
            state.metadata['collapse_mass'] = mass
            return state

    # Occam or fallback
    top_hyp, top_w = _top(state)
    if top_hyp:
        state.metadata = state.metadata if hasattr(state, 'metadata') else {}
        state.metadata['collapse_top'] = top_hyp.id
        state.metadata['collapse_weight'] = top_w

    return state


def apply_collapse(
    state: State,
    mode: CollapseMode,
    selected_hypotheses: List[Hypothesis]
) -> State:
    """Apply collapse decision to state (legacy compatibility).

    Args:
        state: Current state
        mode: Chosen collapse mode
        selected_hypotheses: Hypotheses to keep active

    Returns:
        Updated state after collapse
    """
    if mode == CollapseMode.ZEROBACK:
        # Reset to priors
        new_state = State.from_hypotheses(
            state.hypotheses,
            interpretant_dim=state.interpretant_dim
        )
        new_state.step_num = state.step_num
        new_state.collapse_mode = CollapseMode.ZEROBACK
        return new_state

    elif mode == CollapseMode.OCCAM:
        # Collapse to single hypothesis
        new_posteriors = np.zeros_like(state.posteriors)
        for h in selected_hypotheses:
            idx = state.hypotheses.index(h)
            new_posteriors[idx] = 1.0

        state.posteriors = new_posteriors
        state.collapse_mode = CollapseMode.OCCAM

    elif mode == CollapseMode.HICKAM:
        # Renormalize selected hypotheses
        new_posteriors = np.zeros_like(state.posteriors)
        for h in selected_hypotheses:
            idx = state.hypotheses.index(h)
            new_posteriors[idx] = state.posteriors[idx]

        # Renormalize
        if new_posteriors.sum() > 0:
            new_posteriors = new_posteriors / new_posteriors.sum()

        state.posteriors = new_posteriors
        state.collapse_mode = CollapseMode.HICKAM

    return state
