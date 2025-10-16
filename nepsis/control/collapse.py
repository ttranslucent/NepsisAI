"""Collapse governor for decision-making."""

import numpy as np
from numpy.typing import NDArray

from nepsis.core.types import State, CollapseMode, Hypothesis


def decide_collapse(
    state: State,
    occam_threshold: float = 0.7,
    hickam_threshold: float = 0.3,
    zeroback_threshold: float = 0.8
) -> tuple[CollapseMode, list[Hypothesis]]:
    """Decide which collapse mode to use.

    Three modes:
    1. OCCAM: Single hypothesis (low contradiction, high confidence)
    2. HICKAM: Multiple hypotheses (domain expects multiple causes)
    3. ZEROBACK: Epistemic reset (stuck in contradiction)

    Args:
        state: Current reasoning state
        occam_threshold: Min posterior for Occam collapse
        hickam_threshold: Min posterior for Hickam inclusion
        zeroback_threshold: Max contradiction for avoiding ZeroBack

    Returns:
        Tuple of (collapse_mode, selected_hypotheses)
    """
    max_posterior = state.posteriors.max()
    rho = state.contradiction_density

    # ZeroBack: too much contradiction, need reset
    if rho > zeroback_threshold:
        return CollapseMode.ZEROBACK, []

    # Occam: single clear winner
    if max_posterior >= occam_threshold and rho < 0.3:
        top_idx = np.argmax(state.posteriors)
        return CollapseMode.OCCAM, [state.hypotheses[top_idx]]

    # Hickam: multiple plausible hypotheses
    selected_indices = np.where(state.posteriors >= hickam_threshold)[0]
    if len(selected_indices) > 1:
        selected = [state.hypotheses[i] for i in selected_indices]
        return CollapseMode.HICKAM, selected

    # Default to Occam with top hypothesis
    top_idx = np.argmax(state.posteriors)
    return CollapseMode.OCCAM, [state.hypotheses[top_idx]]


def apply_collapse(
    state: State,
    mode: CollapseMode,
    selected_hypotheses: list[Hypothesis]
) -> State:
    """Apply collapse decision to state.

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
