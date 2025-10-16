"""Core reasoning kernel for NepsisAI."""

from typing import Optional
import numpy as np

from nepsis.core.types import (
    State,
    Signal,
    Hypothesis,
    ReasoningResult,
    CollapseMode,
)
from nepsis.channels.red import check_red_preempt, compute_ruin_probability
from nepsis.channels.blue import process_blue_channel
from nepsis.control.lyapunov import compute_lyapunov, check_convergence, LyapunovWeights
from nepsis.control.collapse import decide_collapse
from nepsis.utils.logging import AuditTrail, StepRecord


def step(
    state: State,
    signal: Signal,
    audit: Optional[AuditTrail] = None,
    lyapunov_weights: Optional[LyapunovWeights] = None,
    ruin_prob: float = 0.0
) -> tuple[State, float]:
    """Execute one reasoning step.

    This is the core reasoning loop:
    1. Check red channel for safety pre-emption
    2. Process through blue channel (interpretant reasoning)
    3. Compute Lyapunov stability
    4. Update ruin probability
    5. Log to audit trail

    Args:
        state: Current reasoning state
        signal: Input signal
        audit: Optional audit trail
        lyapunov_weights: Lyapunov function weights
        ruin_prob: Current ruin probability

    Returns:
        Tuple of (updated_state, updated_ruin_probability)
    """
    # Step 1: Red channel check
    if check_red_preempt(signal):
        state.red_preempted = True
        ruin_prob = compute_ruin_probability(state, ruin_prob, signal)

        # Log red pre-emption
        if audit is not None:
            top_hyp = state.get_top_hypothesis(1)[0]
            record = StepRecord(
                step_num=state.step_num,
                signal_type=str(signal.type),
                signal_name=signal.name,
                signal_value=signal.value,
                contradiction_density=state.contradiction_density,
                lyapunov_value=state.lyapunov_value,
                top_hypothesis=top_hyp.name,
                top_posterior=state.posteriors.max(),
                red_preempted=True,
            )
            audit.add_step(record)

        return state, ruin_prob

    # Step 2: Blue channel processing
    state = process_blue_channel(state, signal)

    # Step 3: Compute Lyapunov stability
    lyapunov_value = compute_lyapunov(state, lyapunov_weights)
    state.lyapunov_value = lyapunov_value
    state.lyapunov_stable = check_convergence(state)

    # Step 4: Update ruin probability
    ruin_prob = compute_ruin_probability(state, ruin_prob, signal)

    # Step 5: Audit logging
    if audit is not None:
        top_hyp = state.get_top_hypothesis(1)[0]
        record = StepRecord(
            step_num=state.step_num,
            signal_type=str(signal.type),
            signal_name=signal.name,
            signal_value=signal.value,
            contradiction_density=state.contradiction_density,
            lyapunov_value=lyapunov_value,
            top_hypothesis=top_hyp.name,
            top_posterior=state.posteriors.max(),
            red_preempted=False,
        )
        audit.add_step(record)

    return state, ruin_prob


def reason(
    signals: list[Signal],
    hypotheses: list[Hypothesis],
    max_steps: int = 100,
    auto_collapse: bool = True,
    audit: Optional[AuditTrail] = None,
    lyapunov_weights: Optional[LyapunovWeights] = None,
) -> ReasoningResult:
    """Execute complete reasoning process.

    Main entry point for NepsisAI reasoning.

    Args:
        signals: List of input signals
        hypotheses: List of hypotheses to reason over
        max_steps: Maximum reasoning steps
        auto_collapse: Whether to auto-collapse when converged
        audit: Optional audit trail for logging
        lyapunov_weights: Lyapunov function weights

    Returns:
        ReasoningResult with final state and decision
    """
    # Initialize state
    state = State.from_hypotheses(hypotheses)

    # Initialize ruin probability
    ruin_prob = 0.0

    # Process each signal
    for i, signal in enumerate(signals):
        if i >= max_steps:
            break

        state, ruin_prob = step(state, signal, audit, lyapunov_weights, ruin_prob)

        # Check for convergence
        if auto_collapse and state.lyapunov_stable:
            break

    # Final collapse decision
    if auto_collapse:
        collapse_mode, selected = decide_collapse(state)
        state.collapse_mode = collapse_mode

    # Build result
    top_hyp = state.get_top_hypothesis(1)[0]
    top_posterior = state.posteriors.max()

    result = ReasoningResult(
        state=state,
        top_hypothesis=top_hyp,
        top_posterior=float(top_posterior),
        contradiction_density=state.contradiction_density,
        converged=state.lyapunov_stable,
        red_preempted=state.red_preempted,
        steps=state.step_num,
        audit_trail=audit.to_dict() if audit else None,
    )

    return result
