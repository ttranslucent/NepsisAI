"""Mathematical utility functions."""

import numpy as np
from numpy.typing import NDArray


def softmax(x: NDArray[np.float32], temperature: float = 1.0) -> NDArray[np.float32]:
    """Compute softmax with optional temperature.

    Args:
        x: Input array
        temperature: Temperature parameter (higher = more uniform)

    Returns:
        Softmax probabilities
    """
    x_scaled = x / temperature
    exp_x = np.exp(x_scaled - np.max(x_scaled))  # Numerical stability
    return exp_x / exp_x.sum()


def normalize(x: NDArray[np.float32], eps: float = 1e-10) -> NDArray[np.float32]:
    """Normalize array to sum to 1.

    Args:
        x: Input array
        eps: Small constant to prevent division by zero

    Returns:
        Normalized array
    """
    total = x.sum()
    if total < eps:
        return np.ones_like(x) / len(x)
    return x / total


def entropy(p: NDArray[np.float32], eps: float = 1e-10) -> float:
    """Compute Shannon entropy.

    Args:
        p: Probability distribution
        eps: Small constant to prevent log(0)

    Returns:
        Entropy value
    """
    p_safe = np.clip(p, eps, 1.0)
    return float(-np.sum(p_safe * np.log(p_safe)))


def kl_divergence(
    p: NDArray[np.float32],
    q: NDArray[np.float32],
    eps: float = 1e-10
) -> float:
    """Compute KL divergence D(p||q).

    Args:
        p: True distribution
        q: Approximate distribution
        eps: Small constant to prevent log(0)

    Returns:
        KL divergence
    """
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))
