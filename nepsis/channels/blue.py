"""Blue channel: Normal interpretant-driven reasoning."""

import numpy as np
from numpy.typing import NDArray

from nepsis.core.types import Signal, State
from nepsis.core.interpretant import apply_interpretant, coherence_score
from nepsis.core.contradiction import compute_contradiction_density
from nepsis.utils.math import normalize


def compute_likelihood(signal: Signal, hypothesis_id: str) -> float:
    """Compute likelihood P(signal | hypothesis).

    This is a simplified likelihood function.
    In a real system, this would be domain-specific.

    Args:
        signal: Observed signal
        hypothesis_id: Hypothesis to evaluate

    Returns:
        Likelihood value
    """
    # Simplified: higher signal values = higher likelihood
    # In reality, each hypothesis would have specific likelihood models
    base_likelihood = min(1.0, signal.value / 10.0)  # Normalize to [0,1]

    # Add some hypothesis-specific variation
    variation = hash(hypothesis_id) % 100 / 100.0
    likelihood = base_likelihood * (0.5 + variation)

    return max(0.01, min(1.0, likelihood))


def process_blue_channel(
    state: State,
    signal: Signal,
    lambda_coherence: float = 1.0
) -> State:
    """Process signal through blue channel (normal reasoning).

    Implements full triadic update:
    1. Interpretant activation
    2. Evidence modulation
    3. Coherence scoring
    4. Bayesian update with coherence
    5. Contradiction check

    Args:
        state: Current reasoning state
        signal: Input signal
        lambda_coherence: Coherence weighting parameter

    Returns:
        Updated state
    """
    # Step 1: Apply interpretant modulation
    modulated_signal, new_interpretant = apply_interpretant(state, signal)

    # Step 2: Compute likelihoods for each hypothesis
    likelihoods = np.array([
        compute_likelihood(signal, h.id) for h in state.hypotheses
    ], dtype=np.float32)

    # Step 3: Compute coherence scores
    coherence_scores = coherence_score(state, likelihoods)

    # Step 4: Bayesian update with coherence modulation
    # π_post ∝ π_prior * L * coh^λ
    posteriors = state.priors * likelihoods * (coherence_scores ** lambda_coherence)
    posteriors = normalize(posteriors)

    # Step 5: Compute contradiction density
    state.posteriors = posteriors
    state.interpretant_states = new_interpretant
    state.likelihoods = likelihoods
    state.coherence_scores = coherence_scores

    rho = compute_contradiction_density(state)
    state.contradiction_density = rho

    # Update history
    state.posterior_history.append(posteriors.copy())
    state.step_num += 1

    # Update priors for next iteration (belief propagation)
    state.priors = posteriors.copy()

    return state
