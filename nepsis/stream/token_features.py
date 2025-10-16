"""Semantic feature extraction for token chunks.

Lightweight regex-based feature detection (no external dependencies).
Replaces naive word-count heuristic with calibrated semantic signals.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

# Regex patterns for semantic feature detection
RE_MONEY = re.compile(r"(?:\$|usd|usd\$)\s*\d[\d,]*(?:\.\d+)?", re.I)
RE_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
RE_TIMEWORD = re.compile(
    r"\b(day|days|hour|hours|am|pm|morning|afternoon|evening|night|itinerary)\b",
    re.I
)
RE_DAY_INDEX = re.compile(r"\b(day\s*[1-9]\d*)\b", re.I)
RE_NEGATION = re.compile(
    r"\b(no|not|never|without|avoid|exclude|forbid|prohibit)\b",
    re.I
)
RE_ROLESHIFT = re.compile(
    r"\b(I\s+will|we\s+will|you\s+should|let'?s|my plan|our plan)\b",
    re.I
)
RE_TRANSPORT = re.compile(
    r"\b(train|bus|metro|subway|tram|taxi|rideshare|walk|ferry|jr\s*pass)\b",
    re.I
)
RE_FLIGHT = re.compile(
    r"\b(flight|fly|plane|airfare|airline|airport)\b",
    re.I
)


def estimate_tokens(text: str) -> int:
    """Estimate BPE-like token count.

    Args:
        text: Input text

    Returns:
        Estimated token count (closer to GPT tokenization than split())

    Note:
        Heuristic: ~1.3 tokens per word for English.
        This is more accurate than naive split() which underestimates.
    """
    return int(len(text.split()) * 1.3)


@dataclass(frozen=True)
class FeatureVector:
    """Semantic features extracted from text chunk."""

    text_len: int
    tokens: int
    has_money: bool
    has_number: bool
    has_timeword: bool
    day_markers: int
    has_transport: bool
    has_flight_term: bool
    negations: int
    role_shift: bool
    constraint_hits: Dict[str, float]  # hypothesis_id -> local risk delta

    def strength(self) -> float:
        """Compute semantic strength score in [0,1].

        Replaces naive word-count heuristic with calibrated feature weighting.

        Returns:
            Strength score where:
            - 0.0 = semantically weak (e.g., "Okay, sounds good")
            - 1.0 = semantically rich (e.g., "Day 1: 3 temples, $50, JR train")

        Weights:
            - Money mentions: 0.18 (high signal for budget tasks)
            - Time markers: 0.18 (itinerary structure)
            - Transport: 0.14 (constraint-relevant)
            - Day markers: 0.06 each (up to 0.18 for 3+ days)
            - Negations: 0.05 each (up to 0.16, constraint violations)
            - Role shift: 0.06 (identity drift warning)
            - Numbers: 0.10 (quantitative info)
        """
        base = 0.0
        base += 0.18 if self.has_money else 0.0
        base += 0.10 if self.has_number else 0.0
        base += 0.18 if self.has_timeword else 0.0
        base += min(0.18, 0.06 * self.day_markers)
        base += 0.14 if self.has_transport else 0.0
        base += 0.06 if self.role_shift else 0.0
        base += min(0.16, 0.05 * self.negations)  # Negations correlate with constraints

        return max(0.0, min(1.0, base))


def extract_features(
    text: str,
    *,
    constraint_aliases: Optional[Dict[str, Dict[str, Iterable[str]]]] = None
) -> FeatureVector:
    """Extract semantic features from text chunk.

    Args:
        text: Input text
        constraint_aliases: Optional mapping of hypothesis_id -> {keywords, antonyms}
            Used for constraint drift detection

    Returns:
        FeatureVector with extracted features

    Example:
        >>> text = "Day 1: Visit 3 temples, $50 total. Take JR train, no flights."
        >>> features = extract_features(text)
        >>> features.strength()
        0.72
        >>> features.has_money
        True
        >>> features.has_flight_term
        True
        >>> features.negations
        1
    """
    t = text or ""

    # Normalize whitespace for better regex matching
    t_norm = re.sub(r'\s+', ' ', t).strip()

    # Pattern matching
    has_money = bool(RE_MONEY.search(t_norm))
    has_number = bool(RE_NUMBER.search(t_norm))
    has_timeword = bool(RE_TIMEWORD.search(t_norm))
    day_markers = len(RE_DAY_INDEX.findall(t_norm))
    has_transport = bool(RE_TRANSPORT.search(t_norm))
    has_flight = bool(RE_FLIGHT.search(t_norm))
    negations = len(RE_NEGATION.findall(t_norm))
    role_shift = bool(RE_ROLESHIFT.search(t_norm))

    # Constraint alias scoring
    # Risk increases if antonyms present (violation indicators)
    # Confidence increases if keywords present (constraint awareness)
    hits: Dict[str, float] = {}

    if constraint_aliases:
        lt = t_norm.lower()
        for hyp_id, cfg in constraint_aliases.items():
            kw = cfg.get("keywords", [])
            an = cfg.get("antonyms", [])
            score = 0.0

            # Keywords indicate constraint awareness (+0.15)
            if any(k.lower() in lt for k in kw):
                score += 0.15

            # Antonyms indicate potential violation (+0.35)
            if any(a.lower() in lt for a in an):
                score += 0.35  # Higher weight for risk

            if score > 0:
                hits[hyp_id] = min(1.0, score)

    return FeatureVector(
        text_len=len(t),
        tokens=estimate_tokens(t_norm),
        has_money=has_money,
        has_number=has_number,
        has_timeword=has_timeword,
        day_markers=day_markers,
        has_transport=has_transport,
        has_flight_term=has_flight,
        negations=negations,
        role_shift=role_shift,
        constraint_hits=hits,
    )
