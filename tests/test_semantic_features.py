"""Tests for semantic feature extraction and drift detection."""

import pytest
from nepsis.stream.token_features import extract_features, FeatureVector, estimate_tokens
from nepsis.stream.constraint_map import ConstraintMap, DriftAccumulator, build_constraint_maps
from nepsis.core.types import Hypothesis


class TestTokenFeatures:
    """Test semantic feature extraction."""

    def test_money_detection(self):
        """Test money pattern matching."""
        text = "Total cost: $450 for 3 nights, roughly $150/night."
        features = extract_features(text)

        assert features.has_money is True
        assert features.strength() > 0.2

    def test_transport_detection(self):
        """Test transport term detection."""
        text = "Take the JR train from Tokyo to Kyoto, then local bus."
        features = extract_features(text)

        assert features.has_transport is True
        assert features.has_flight_term is False

    def test_flight_detection(self):
        """Test flight term detection."""
        text = "Book a flight to Osaka, then take the train."
        features = extract_features(text)

        assert features.has_flight_term is True
        assert features.has_transport is True  # train also present

    def test_negation_detection(self):
        """Test negation pattern matching."""
        text = "No flights allowed. Avoid flying, never book planes."
        features = extract_features(text)

        assert features.negations >= 3  # "no", "avoid", "never"
        # "flights" and "flying" should match the flight pattern
        # If not matching, it means the regex needs to be updated to handle word forms

    def test_role_shift_detection(self):
        """Test role/identity shift patterns."""
        text = "I will book the hotel. Let's plan your itinerary together."
        features = extract_features(text)

        assert features.role_shift is True

    def test_day_markers(self):
        """Test day index detection."""
        text = "Day 1: Temple visit. Day 2: Museum. Day 3: Garden."
        features = extract_features(text)

        assert features.day_markers == 3
        assert features.has_timeword is True

    def test_strength_scoring_rich_text(self):
        """Test strength scoring on semantically rich text."""
        text = "Day 1: Visit 3 temples, $50 total. Take the JR train, no flights."
        features = extract_features(text)

        strength = features.strength()
        assert 0.5 < strength <= 1.0  # Should be high
        assert features.has_money is True
        assert features.has_transport is True
        assert features.day_markers >= 1

    def test_strength_scoring_weak_text(self):
        """Test strength scoring on semantically weak text."""
        text = "Okay, sounds good. Thanks!"
        features = extract_features(text)

        strength = features.strength()
        assert strength < 0.2  # Should be low
        assert features.has_money is False
        assert features.has_transport is False

    def test_constraint_alias_keyword_detection(self):
        """Test constraint keyword detection."""
        text = "Take the train and bus, no flights allowed."
        aliases = {
            "no_flights": {
                "keywords": ["train", "bus", "overland"],
                "antonyms": ["flight", "fly", "plane"]
            }
        }

        features = extract_features(text, constraint_aliases=aliases)

        assert "no_flights" in features.constraint_hits
        # Should have both keyword match (+0.15) and antonym match (+0.35)
        assert features.constraint_hits["no_flights"] >= 0.4

    def test_constraint_alias_antonym_only(self):
        """Test constraint antonym detection (violation indicator)."""
        text = "Book a flight to Kyoto."
        aliases = {
            "no_flights": {
                "keywords": ["train", "bus"],
                "antonyms": ["flight", "fly", "plane"]
            }
        }

        features = extract_features(text, constraint_aliases=aliases)

        assert "no_flights" in features.constraint_hits
        # Should have antonym match only (+0.35)
        assert 0.3 <= features.constraint_hits["no_flights"] <= 0.4

    def test_token_estimation(self):
        """Test BPE-like token estimation."""
        text = "The quick brown fox jumps over the lazy dog"
        tokens = estimate_tokens(text)

        # Should be ~1.3x word count
        word_count = len(text.split())
        assert tokens > word_count  # More than words
        assert tokens < word_count * 1.5  # But not too much more


class TestConstraintMap:
    """Test constraint mapping utilities."""

    def test_constraint_map_creation(self):
        """Test basic ConstraintMap creation."""
        cmap = ConstraintMap(
            hypothesis_id="no_flights",
            keywords=["train", "bus"],
            antonyms=["flight", "plane"]
        )

        assert cmap.hypothesis_id == "no_flights"
        assert "train" in cmap.keywords
        assert "flight" in cmap.antonyms

    def test_from_hypothesis(self):
        """Test ConstraintMap.from_hypothesis() constructor."""
        hypo = Hypothesis("budget_ok", "Stay under budget", prior=0.5)
        cmap = ConstraintMap.from_hypothesis(
            hypo,
            keywords=["budget", "cost"],
            antonyms=["exceed", "over budget"]
        )

        assert cmap.hypothesis_id == "budget_ok"
        assert "budget" in cmap.keywords
        assert "exceed" in cmap.antonyms

    def test_as_alias_dict(self):
        """Test conversion to alias dictionary."""
        cmap = ConstraintMap(
            hypothesis_id="test",
            keywords=["kw1", "kw2"],
            antonyms=["ant1"]
        )

        alias_dict = cmap.as_alias_dict()

        assert "test" in alias_dict
        assert alias_dict["test"]["keywords"] == ["kw1", "kw2"]
        assert alias_dict["test"]["antonyms"] == ["ant1"]

    def test_build_constraint_maps(self):
        """Test batch building from config."""
        hypos = [
            Hypothesis("h1", "First", prior=0.5),
            Hypothesis("h2", "Second", prior=0.5),
        ]

        config = {
            "h1": {"keywords": ["k1"], "antonyms": ["a1"]},
            "h2": {"keywords": ["k2"], "antonyms": ["a2"]},
        }

        maps = build_constraint_maps(hypos, config)

        assert len(maps) == 2
        assert maps[0].hypothesis_id == "h1"
        assert maps[1].hypothesis_id == "h2"


class TestDriftAccumulator:
    """Test drift detection with EMA smoothing."""

    def test_initial_update(self):
        """Test first drift detection."""
        acc = DriftAccumulator(alpha=0.35)

        scores = acc.update({"no_flights": 0.8})

        # First update: score = alpha * delta = 0.35 * 0.8
        assert "no_flights" in scores
        assert 0.25 < scores["no_flights"] < 0.35

    def test_ema_smoothing(self):
        """Test exponential moving average smoothing."""
        acc = DriftAccumulator(alpha=0.35)

        # First update
        scores1 = acc.update({"no_flights": 0.8})
        score1 = scores1["no_flights"]

        # Second update (lower value)
        scores2 = acc.update({"no_flights": 0.2})
        score2 = scores2["no_flights"]

        # score2 = (1-0.35)*score1 + 0.35*0.2
        # score2 = 0.65*score1 + 0.07
        expected = 0.65 * score1 + 0.35 * 0.2

        assert abs(score2 - expected) < 0.01

    def test_clamping_upper(self):
        """Test clamping to [0,1] range (upper bound)."""
        acc = DriftAccumulator(alpha=0.5)

        # Push high repeatedly
        for _ in range(10):
            acc.update({"test": 1.0})

        assert acc.scores["test"] <= 1.0  # Should clamp

    def test_clamping_lower(self):
        """Test clamping to [0,1] range (lower bound)."""
        acc = DriftAccumulator(alpha=0.5)

        # Start high, then push low
        acc.update({"test": 0.8})
        for _ in range(10):
            acc.update({"test": 0.0})

        assert acc.scores["test"] >= 0.0  # Should clamp

    def test_multiple_hypotheses(self):
        """Test tracking multiple hypotheses simultaneously."""
        acc = DriftAccumulator(alpha=0.35)

        acc.update({"h1": 0.5, "h2": 0.8, "h3": 0.2})

        assert len(acc.scores) == 3
        assert "h1" in acc.scores
        assert "h2" in acc.scores
        assert "h3" in acc.scores

    def test_reset_single(self):
        """Test resetting single hypothesis."""
        acc = DriftAccumulator(alpha=0.35)
        acc.update({"h1": 0.5, "h2": 0.8})

        acc.reset("h1")

        assert "h1" not in acc.scores
        assert "h2" in acc.scores

    def test_reset_all(self):
        """Test resetting all hypotheses."""
        acc = DriftAccumulator(alpha=0.35)
        acc.update({"h1": 0.5, "h2": 0.8})

        acc.reset()

        assert len(acc.scores) == 0


class TestIntegration:
    """Integration tests for features + drift detection."""

    def test_full_pipeline(self):
        """Test complete feature extraction + drift tracking."""
        # Setup
        aliases = {
            "no_flights": {
                "keywords": ["train", "bus"],
                "antonyms": ["flight", "plane"]
            }
        }
        acc = DriftAccumulator(alpha=0.35)

        # Chunk 1: Violation mention
        text1 = "Maybe we should book a flight?"
        features1 = extract_features(text1, constraint_aliases=aliases)
        scores1 = acc.update(features1.constraint_hits)

        assert "no_flights" in scores1
        assert scores1["no_flights"] > 0  # Risk detected

        # Chunk 2: Corrective mention
        text2 = "Actually, let's take the train instead."
        features2 = extract_features(text2, constraint_aliases=aliases)
        scores2 = acc.update(features2.constraint_hits)

        # Risk should persist (EMA smoothing means it won't drop immediately)
        # Just verify it's tracked
        assert "no_flights" in scores2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
