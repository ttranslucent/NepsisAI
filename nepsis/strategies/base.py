"""Base strategy configuration."""

from dataclasses import dataclass
from typing import Dict
from nepsis.control.lyapunov import LyapunovWeights


@dataclass
class StrategyConfig:
    """Configuration for domain-specific reasoning strategy."""

    name: str
    collapse_mode: str = "occam"  # "occam", "hickam", or "auto"
    red_channel_threshold: float = 0.8
    lyapunov_weights: Dict[str, float] | None = None
    occam_threshold: float = 0.7
    hickam_threshold: float = 0.3
    zeroback_threshold: float = 0.8

    def get_lyapunov_weights(self) -> LyapunovWeights:
        """Get Lyapunov weights configuration."""
        if self.lyapunov_weights is None:
            return LyapunovWeights()

        return LyapunovWeights(
            contradiction=self.lyapunov_weights.get("contradiction", 2.0),
            entropy=self.lyapunov_weights.get("entropy", 1.0),
            coherence=self.lyapunov_weights.get("coherence", 1.5),
            velocity=self.lyapunov_weights.get("velocity", 0.5),
        )


class Strategy:
    """Base class for reasoning strategies."""

    def __init__(self, config: StrategyConfig):
        """Initialize strategy with configuration.

        Args:
            config: Strategy configuration
        """
        self.config = config

    @classmethod
    def emergency_medicine(cls) -> "Strategy":
        """Emergency medicine strategy.

        Characteristics:
        - High red channel sensitivity
        - Hickam mode (multiple diagnoses expected)
        - High coherence weight (pattern recognition critical)
        """
        config = StrategyConfig(
            name="emergency_medicine",
            collapse_mode="hickam",
            red_channel_threshold=0.7,
            lyapunov_weights={
                "contradiction": 2.5,
                "entropy": 1.0,
                "coherence": 2.0,
                "velocity": 0.3,
            },
            hickam_threshold=0.25,
        )
        return cls(config)

    @classmethod
    def research(cls) -> "Strategy":
        """Research/exploration strategy.

        Characteristics:
        - Lower red channel threshold (tolerate uncertainty)
        - High entropy tolerance (explore hypotheses)
        - Slower convergence
        """
        config = StrategyConfig(
            name="research",
            collapse_mode="auto",
            red_channel_threshold=0.9,
            lyapunov_weights={
                "contradiction": 1.5,
                "entropy": 2.0,
                "coherence": 1.0,
                "velocity": 0.8,
            },
            occam_threshold=0.8,
        )
        return cls(config)

    @classmethod
    def default(cls) -> "Strategy":
        """Default balanced strategy."""
        config = StrategyConfig(
            name="default",
            collapse_mode="occam",
        )
        return cls(config)
