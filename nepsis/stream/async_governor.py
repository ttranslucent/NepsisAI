"""Async governor for real-time stream state management.

Processes semantic chunks asynchronously and maintains Nepsis reasoning state.
"""

from __future__ import annotations
from typing import Optional, List
from dataclasses import dataclass
import numpy as np

from nepsis.core.types import State, Signal, Hypothesis, SignalType
from nepsis.core.kernel import step as nepsis_step
from nepsis.control.lyapunov import LyapunovWeights
from nepsis.stream.token_buffer import TokenChunk
from nepsis.stream.token_features import extract_features
from nepsis.stream.constraint_map import ConstraintMap, DriftAccumulator


@dataclass
class GovernorMetrics:
    """Metrics from governor processing."""
    token_index: int
    fidelity_score: float
    contradiction_density: float
    interpretant_coherence: float
    constraint_risk: dict[str, float]
    collapse_suggest: str  # 'none', 'hickam', 'zeroback'
    lyapunov_value: float
    ruin_prob: float


class StreamGovernor:
    """Governs LLM stream with Nepsis reasoning logic.

    Processes semantic chunks and maintains state in real-time.
    """

    def __init__(
        self,
        hypotheses: List[Hypothesis],
        interpretant_dim: int = 16,
        lyapunov_weights: Optional[LyapunovWeights] = None,
        constraint_maps: Optional[List[ConstraintMap]] = None
    ):
        """Initialize stream governor.

        Args:
            hypotheses: Hypotheses to track
            interpretant_dim: Dimensionality of interpretant state
            lyapunov_weights: Lyapunov function weights
            constraint_maps: Optional constraint maps for drift detection
        """
        self.state = State.from_hypotheses(
            hypotheses,
            interpretant_dim=interpretant_dim
        )
        self.lyapunov_weights = lyapunov_weights
        self.ruin_prob = 0.0
        self.chunk_count = 0

        # Constraint drift tracking (NEW)
        self.constraint_maps = constraint_maps or []
        self._drift = DriftAccumulator(alpha=0.35)

        # Build alias dict for feature extraction
        self._alias_dict = {}
        for cmap in self.constraint_maps:
            self._alias_dict.update(cmap.as_alias_dict())

    def process_chunk(self, chunk: TokenChunk) -> GovernorMetrics:
        """Process semantic chunk and update state.

        Args:
            chunk: Semantic token chunk from buffer

        Returns:
            GovernorMetrics with current reasoning state
        """
        # Convert chunk to Signal
        signal = self._chunk_to_signal(chunk)

        # Run Nepsis reasoning step
        self.state, self.ruin_prob = nepsis_step(
            state=self.state,
            signal=signal,
            audit=None,  # No audit trail in streaming mode
            lyapunov_weights=self.lyapunov_weights,
            ruin_prob=self.ruin_prob
        )

        self.chunk_count += 1

        # Compute metrics
        metrics = self._extract_metrics(chunk)

        return metrics

    def _chunk_to_signal(self, chunk: TokenChunk) -> Signal:
        """Convert token chunk to Nepsis Signal.

        Args:
            chunk: Token chunk

        Returns:
            Signal object
        """
        # Semantic signal extraction
        # For v0.3.0: Use text features as signal value
        signal_value = self._compute_signal_value(chunk)

        return Signal(
            type=SignalType.HISTORY,  # Chunks are history/context
            name=f"chunk_{self.chunk_count}",
            value=signal_value,
            metadata={
                "text": chunk.text,
                "is_complete": chunk.is_complete,
                "token_count": chunk.token_count,
                "start_idx": chunk.start_idx,
                "end_idx": chunk.end_idx
            }
        )

    def _compute_signal_value(self, chunk: TokenChunk) -> float:
        """Compute signal value from semantic features.

        Args:
            chunk: Token chunk

        Returns:
            Signal strength value [0, 1]

        Note:
            Uses semantic feature pack (money, transport, time, negation, etc.)
            to compute calibrated strength score. Stores features in metadata.
        """
        # Extract semantic features
        features = extract_features(
            chunk.text,
            constraint_aliases=self._alias_dict if self._alias_dict else None
        )

        # Update drift accumulator
        if features.constraint_hits:
            drift_scores = self._drift.update(features.constraint_hits)

        # Return calibrated strength
        return features.strength()

    def _extract_metrics(self, chunk: TokenChunk) -> GovernorMetrics:
        """Extract metrics from current state.

        Args:
            chunk: Processed chunk

        Returns:
            GovernorMetrics
        """
        # Fidelity score: inverse of contradiction
        fidelity = 1.0 - self.state.contradiction_density

        # Interpretant coherence: mean coherence score
        if len(self.state.coherence_scores) > 0:
            interpretant_coherence = float(np.mean(self.state.coherence_scores))
        else:
            interpretant_coherence = 1.0

        # Constraint risk: combines posteriors with drift scores
        constraint_risk = {
            h.id: float(self.state.posteriors[i])
            for i, h in enumerate(self.state.hypotheses)
        }

        # Add drift scores to constraint risk
        for hyp_id, drift_score in self._drift.scores.items():
            if hyp_id in constraint_risk:
                # Combine posterior and drift (weighted average)
                constraint_risk[hyp_id] = 0.6 * constraint_risk[hyp_id] + 0.4 * drift_score
            else:
                constraint_risk[hyp_id] = drift_score

        # Collapse suggestion
        if self.state.contradiction_density > 0.75:
            collapse_suggest = "zeroback"
        elif self.state.contradiction_density > 0.40:
            collapse_suggest = "hickam"
        else:
            collapse_suggest = "none"

        return GovernorMetrics(
            token_index=chunk.end_idx,
            fidelity_score=fidelity,
            contradiction_density=self.state.contradiction_density,
            interpretant_coherence=interpretant_coherence,
            constraint_risk=constraint_risk,
            collapse_suggest=collapse_suggest,
            lyapunov_value=self.state.lyapunov_value,
            ruin_prob=self.ruin_prob
        )

    def reset(self):
        """Reset governor to initial state (ZeroBack)."""
        self.state = State.from_hypotheses(
            self.state.hypotheses,
            interpretant_dim=self.state.interpretant_dim
        )
        self.ruin_prob = 0.0
        self.chunk_count = 0
