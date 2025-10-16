"""Red channel: Safety-critical signal processing."""

import numpy as np
from nepsis.core.types import Signal, State


def check_red_preempt(signal: Signal) -> bool:
    """Check if signal triggers red channel pre-emption.

    Red signals bypass normal reasoning and trigger immediate safety responses.

    Args:
        signal: Input signal

    Returns:
        True if red channel should pre-empt
    """
    # Check explicit red threshold
    if signal.red_threshold is not None:
        if signal.value >= signal.red_threshold:
            return True

    # Check signal type
    if signal.type == "red":
        return True

    return False


def compute_ruin_probability(
    state: State,
    current_ruin: float = 0.0,
    signal: Signal | None = None
) -> float:
    """Compute cumulative ruin probability (monotone increasing).

    In non-ergodic systems, ruin probability never decreases.
    Each unsafe state contributes to total ruin risk.

    Args:
        state: Current reasoning state
        current_ruin: Previous ruin probability
        signal: New signal (may increase ruin)

    Returns:
        Updated ruin probability
    """
    # Start with current ruin (monotone property)
    new_ruin = current_ruin

    # Check for red signals
    if signal is not None and check_red_preempt(signal):
        # Red signal increases ruin probability
        risk_increment = 0.1  # Per red signal
        new_ruin = min(1.0, current_ruin + risk_increment)

    # High contradiction also increases ruin risk
    if state.contradiction_density > 0.5:
        confusion_risk = (state.contradiction_density - 0.5) * 0.05
        new_ruin = min(1.0, new_ruin + confusion_risk)

    return new_ruin
