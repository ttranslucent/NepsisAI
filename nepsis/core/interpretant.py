"""Interpretant layer for triadic reasoning."""

from __future__ import annotations
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from nepsis.core.types import Signal, State
from nepsis.utils.math import softmax, normalize


def apply_interpretant(
    state: State,
    signal: Signal,
    gamma_matrix: Optional[NDArray[np.float32]] = None
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Apply interpretant modulation to evidence.

    Implements: S_modulated = S ⊙ (I_states @ Γ_I)

    Args:
        state: Current reasoning state
        signal: Input signal
        gamma_matrix: S→I gating matrix (if None, uses identity-like)

    Returns:
        Tuple of (modulated_signal, updated_interpretant_states)
    """
    n_hyp = len(state.hypotheses)
    i_dim = state.interpretant_dim

    # Create default gamma matrix if not provided
    # Maps signal value to interpretant activation pattern
    if gamma_matrix is None:
        # Simple default: distribute signal across interpretants
        gamma_matrix = np.ones((i_dim, n_hyp), dtype=np.float32) / i_dim

    # Update interpretant states based on signal
    # Signal activates certain interpretant dimensions
    signal_vector = np.full(n_hyp, signal.value, dtype=np.float32)

    # Gating: which interpretants get activated by this signal?
    interpretant_update = state.interpretant_states @ gamma_matrix

    # Modulate signal by interpretant state
    modulated_signal = signal_vector * np.mean(interpretant_update, axis=0)

    # Update interpretant states (moving average)
    alpha = 0.3  # Learning rate
    new_interpretant = (1 - alpha) * state.interpretant_states + alpha * normalize(
        state.interpretant_states * signal.value
    )

    return modulated_signal, new_interpretant


def coherence_score(
    state: State,
    likelihoods: NDArray[np.float32],
    compatibility_matrix: Optional[NDArray[np.float32]] = None
) -> NDArray[np.float32]:
    """Compute coherence between interpretant and hypotheses.

    Implements: coh = (I_states @ C_IO) ⊙ softmax(L)

    Args:
        state: Current reasoning state
        likelihoods: Likelihood vector for each hypothesis
        compatibility_matrix: I→O compatibility (if None, uses uniform)

    Returns:
        Coherence scores for each hypothesis
    """
    n_hyp = len(state.hypotheses)
    i_dim = state.interpretant_dim

    # Create default compatibility matrix if not provided
    # How well does each interpretant state support each hypothesis?
    if compatibility_matrix is None:
        # Uniform compatibility as default
        compatibility_matrix = np.ones((i_dim, n_hyp), dtype=np.float32) / i_dim

    # Interpretant contribution to each hypothesis
    interpretant_contrib = state.interpretant_states @ compatibility_matrix

    # Modulate by likelihood softmax (more confident = higher coherence weight)
    likelihood_weights = softmax(likelihoods, temperature=0.5)

    # Coherence is alignment between interpretant and evidence
    coherence = interpretant_contrib * likelihood_weights

    return normalize(coherence)


def triadic_consistency(
    signal_value: float,
    interpretant_states: NDArray[np.float32],
    hypothesis_posteriors: NDArray[np.float32]
) -> float:
    """Check triadic consistency: S-I-O alignment.

    Measures whether signal, interpretant, and hypotheses are mutually coherent.

    Args:
        signal_value: Raw signal value
        interpretant_states: Current interpretant distribution
        hypothesis_posteriors: Current hypothesis beliefs

    Returns:
        Consistency score in [0, 1]
    """
    # Signal strength should align with interpretant activation
    i_strength = np.linalg.norm(interpretant_states)
    s_i_alignment = 1.0 - abs(signal_value - i_strength) / max(signal_value, i_strength, 1e-6)

    # Interpretant entropy should relate to hypothesis entropy
    i_entropy = -np.sum(interpretant_states * np.log(interpretant_states + 1e-10))
    h_entropy = -np.sum(hypothesis_posteriors * np.log(hypothesis_posteriors + 1e-10))
    i_h_alignment = 1.0 - abs(i_entropy - h_entropy) / max(i_entropy, h_entropy, 1e-6)

    # Combined consistency
    consistency = (s_i_alignment + i_h_alignment) / 2.0

    return float(np.clip(consistency, 0.0, 1.0))
