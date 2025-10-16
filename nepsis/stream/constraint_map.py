"""Constraint mapping and drift detection for streaming governance.

Tracks constraint adherence via keyword/antonym matching with EMA smoothing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional
from collections import defaultdict

from nepsis.core.types import Hypothesis


@dataclass
class ConstraintMap:
    """Maps hypothesis to constraint keywords and violation indicators.

    Attributes:
        hypothesis_id: ID of hypothesis this constrains
        keywords: Terms that indicate constraint awareness (positive signal)
        antonyms: Terms that indicate potential violation (risk signal)
    """

    hypothesis_id: str
    keywords: List[str] = field(default_factory=list)
    antonyms: List[str] = field(default_factory=list)

    def as_alias_dict(self) -> Dict[str, Dict[str, Iterable[str]]]:
        """Convert to alias dictionary for feature extraction.

        Returns:
            Dict mapping hypothesis_id -> {keywords, antonyms}
        """
        return {
            self.hypothesis_id: {
                "keywords": self.keywords,
                "antonyms": self.antonyms
            }
        }

    @classmethod
    def from_hypothesis(
        cls,
        hypothesis: Hypothesis,
        keywords: Optional[List[str]] = None,
        antonyms: Optional[List[str]] = None
    ) -> "ConstraintMap":
        """Create ConstraintMap from Hypothesis object.

        Args:
            hypothesis: Hypothesis to map
            keywords: Constraint awareness terms
            antonyms: Violation indicator terms

        Returns:
            ConstraintMap instance
        """
        return cls(
            hypothesis_id=hypothesis.id,
            keywords=keywords or [],
            antonyms=antonyms or []
        )


def build_constraint_maps(
    hypotheses: List[Hypothesis],
    alias_config: Dict[str, Dict[str, List[str]]]
) -> List[ConstraintMap]:
    """Build constraint maps from configuration dictionary.

    Args:
        hypotheses: List of hypotheses
        alias_config: Dict mapping hypothesis_id -> {keywords, antonyms}

    Returns:
        List of ConstraintMap objects

    Example:
        >>> hypos = [Hypothesis("no_flights", "No flights", prior=0.8)]
        >>> config = {
        ...     "no_flights": {
        ...         "keywords": ["train", "bus", "overland"],
        ...         "antonyms": ["flight", "plane", "airfare"]
        ...     }
        ... }
        >>> maps = build_constraint_maps(hypos, config)
    """
    maps = []
    for h in hypotheses:
        if h.id in alias_config:
            cfg = alias_config[h.id]
            maps.append(ConstraintMap(
                hypothesis_id=h.id,
                keywords=cfg.get("keywords", []),
                antonyms=cfg.get("antonyms", [])
            ))
    return maps


@dataclass
class DriftAccumulator:
    """Tracks cumulative constraint risk with EMA smoothing.

    Uses exponential moving average to prevent oscillation from
    noisy single-chunk detections.

    Attributes:
        alpha: EMA smoothing factor [0,1]
            - Higher α = faster adaptation to new signals
            - Lower α = more smoothing, less noise
            - Default 0.35 balances responsiveness and stability
        scores: Current risk scores per hypothesis [0,1]
    """

    alpha: float = 0.35
    scores: Dict[str, float] = field(default_factory=dict)

    def update(self, deltas: Dict[str, float]) -> Dict[str, float]:
        """Update drift scores with new detections.

        Args:
            deltas: New risk deltas from current chunk

        Returns:
            Updated cumulative scores (clamped to [0,1])

        Formula:
            score[t] = (1-α)·score[t-1] + α·delta[t]

        Example:
            >>> acc = DriftAccumulator(alpha=0.35)
            >>> scores = acc.update({"no_flights": 0.8})  # First detection
            >>> # score ≈ 0.35 * 0.8 = 0.28
            >>> scores = acc.update({"no_flights": 0.2})  # Second detection
            >>> # score ≈ 0.65 * 0.28 + 0.35 * 0.2 = 0.25
        """
        for k, d in deltas.items():
            prev = self.scores.get(k, 0.0)
            # EMA update with clamping
            new_score = (1 - self.alpha) * prev + self.alpha * d
            self.scores[k] = max(0.0, min(1.0, new_score))

        return self.scores

    def reset(self, hypothesis_id: Optional[str] = None):
        """Reset drift scores.

        Args:
            hypothesis_id: If provided, reset only this hypothesis.
                          If None, reset all scores.
        """
        if hypothesis_id:
            self.scores.pop(hypothesis_id, None)
        else:
            self.scores.clear()


def default_trip_planner_maps() -> List[ConstraintMap]:
    """Default constraint maps for trip planning tasks.

    Returns:
        List of ConstraintMap for common trip planning constraints

    Covers:
        - no_flights: Flight usage prohibition
        - budget_ok: Budget adherence
        - site_count: Cultural site visit requirements
    """
    return [
        ConstraintMap(
            hypothesis_id="no_flights",
            keywords=["overland", "train", "bus", "JR pass", "local transit", "walk", "ferry"],
            antonyms=["flight", "fly", "plane", "airfare", "airport", "airline", "booking flight"],
        ),
        ConstraintMap(
            hypothesis_id="budget_ok",
            keywords=["budget", "total", "per day", "cost", "price", "under budget", "within budget"],
            antonyms=["over budget", "exceed", "overage", "additional cost", "extra charge"],
        ),
        ConstraintMap(
            hypothesis_id="site_count",
            keywords=["temple", "shrine", "garden", "museum", "palace", "castle", "site"],
            antonyms=[],  # Count-based constraint, handled separately
        ),
    ]


def default_medical_maps() -> List[ConstraintMap]:
    """Default constraint maps for medical reasoning tasks.

    Returns:
        List of ConstraintMap for medical differential diagnosis

    Example hypotheses:
        - no_antibiotics: Antibiotic avoidance (allergy, resistance)
        - stable_vitals: Vital sign stability requirement
    """
    return [
        ConstraintMap(
            hypothesis_id="no_antibiotics",
            keywords=["alternative", "non-antibiotic", "supportive care"],
            antonyms=["antibiotic", "penicillin", "cephalosporin", "prescribe antibiotics"],
        ),
        ConstraintMap(
            hypothesis_id="stable_vitals",
            keywords=["stable", "normal", "within range", "improving"],
            antonyms=["unstable", "deteriorating", "hypotensive", "tachycardic", "dropping"],
        ),
    ]
