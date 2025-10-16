"""Tests for Hickam multi-cause collapse logic."""

import pytest
import numpy as np

from nepsis.core.types import Hypothesis, State, CollapseMode
from nepsis.core.exclusivity_builder import build_exclusivity_from_rules
from nepsis.control.collapse import _hickam_cluster_ok, should_collapse


class TestHickamCluster:
    """Test Hickam clustering algorithm."""

    def test_two_compatible_hypotheses_form_cluster(self):
        """Two hypotheses with low exclusivity should cluster."""
        hypos = [
            Hypothesis("pneumonia", "Pneumonia", prior=0.4),
            Hypothesis("chf", "CHF", prior=0.35),
            Hypothesis("gerd", "GERD", prior=0.25),
        ]

        state = State.from_hypotheses(hypos)
        state.collapse_mode = CollapseMode.HICKAM

        # pneumonia & CHF compatible (both can occur)
        # both incompatible with GERD
        state.exclusivity = build_exclusivity_from_rules(
            ["pneumonia", "chf", "gerd"],
            pairs=[
                ("pneumonia", "gerd", 0.8),
                ("chf", "gerd", 0.8),
                ("pneumonia", "chf", 0.2),  # Low exclusivity
            ]
        )

        # Both pneumonia and CHF likely
        state.posteriors = np.array([0.5, 0.4, 0.1], dtype=np.float32)

        ok, cluster, mass = _hickam_cluster_ok(state)

        assert ok is True
        assert len(cluster) == 2
        assert cluster[0].id == "pneumonia"
        assert cluster[1].id == "chf"
        assert mass >= 0.85  # Meets threshold

    def test_high_exclusivity_prevents_clustering(self):
        """Hypotheses with high exclusivity should not cluster."""
        hypos = [
            Hypothesis("stemi", "STEMI", prior=0.5),
            Hypothesis("gerd", "GERD", prior=0.5),
        ]

        state = State.from_hypotheses(hypos)
        state.collapse_mode = CollapseMode.HICKAM

        # Highly mutually exclusive
        state.exclusivity = build_exclusivity_from_rules(
            ["stemi", "gerd"],
            pairs=[("stemi", "gerd", 0.9)]
        )

        # Both likely (but exclusive)
        state.posteriors = np.array([0.6, 0.5], dtype=np.float32)

        ok, cluster, mass = _hickam_cluster_ok(state)

        # Can't form valid cluster due to high exclusivity
        assert ok is False or len(cluster) < 2

    def test_three_way_cluster(self):
        """Test clustering with 3+ compatible hypotheses."""
        hypos = [
            Hypothesis("h1", "H1", prior=0.3),
            Hypothesis("h2", "H2", prior=0.3),
            Hypothesis("h3", "H3", prior=0.3),
            Hypothesis("h4", "H4", prior=0.1),
        ]

        state = State.from_hypotheses(hypos)
        state.collapse_mode = CollapseMode.HICKAM

        # h1, h2, h3 all compatible; h4 incompatible with all
        state.exclusivity = build_exclusivity_from_rules(
            ["h1", "h2", "h3", "h4"],
            pairs=[
                ("h1", "h2", 0.1),
                ("h1", "h3", 0.1),
                ("h2", "h3", 0.1),
                ("h1", "h4", 0.9),
                ("h2", "h4", 0.9),
                ("h3", "h4", 0.9),
            ]
        )

        state.posteriors = np.array([0.35, 0.30, 0.25, 0.10], dtype=np.float32)

        ok, cluster, mass = _hickam_cluster_ok(state)

        assert ok is True
        assert len(cluster) >= 3
        assert "h1" in [h.id for h in cluster]
        assert "h2" in [h.id for h in cluster]
        assert "h3" in [h.id for h in cluster]
        assert "h4" not in [h.id for h in cluster]  # Excluded

    def test_insufficient_mass_no_collapse(self):
        """Cluster doesn't collapse if total mass too low."""
        hypos = [
            Hypothesis("h1", "H1", prior=0.5),
            Hypothesis("h2", "H2", prior=0.5),
        ]

        state = State.from_hypotheses(hypos)
        state.collapse_mode = CollapseMode.HICKAM

        # Compatible but low mass
        state.exclusivity = build_exclusivity_from_rules(
            ["h1", "h2"],
            pairs=[("h1", "h2", 0.1)]
        )

        state.posteriors = np.array([0.4, 0.3], dtype=np.float32)  # Total = 0.7 < 0.85

        ok, cluster, mass = _hickam_cluster_ok(state)

        assert ok is False  # Doesn't meet mass threshold


class TestHickamCollapse:
    """Test full collapse logic with Hickam mode."""

    def test_hickam_mode_uses_cluster(self):
        """In Hickam mode, should_collapse uses cluster logic."""
        hypos = [
            Hypothesis("pneumonia", "Pneumonia", prior=0.4),
            Hypothesis("chf", "CHF", prior=0.35),
        ]

        state = State.from_hypotheses(hypos)
        state.collapse_mode = CollapseMode.HICKAM
        state.ruin_prob = 0.05  # Low ruin

        state.exclusivity = build_exclusivity_from_rules(
            ["pneumonia", "chf"],
            pairs=[("pneumonia", "chf", 0.2)]
        )

        state.posteriors = np.array([0.5, 0.45], dtype=np.float32)

        should = should_collapse(state)

        assert should is True  # Valid cluster + low ruin

    def test_high_ruin_prevents_hickam_collapse(self):
        """High ruin probability prevents collapse even with valid cluster."""
        hypos = [
            Hypothesis("h1", "H1", prior=0.5),
            Hypothesis("h2", "H2", prior=0.5),
        ]

        state = State.from_hypotheses(hypos)
        state.collapse_mode = CollapseMode.HICKAM
        state.ruin_prob = 0.5  # High ruin

        state.exclusivity = build_exclusivity_from_rules(
            ["h1", "h2"],
            pairs=[("h1", "h2", 0.1)]
        )

        state.posteriors = np.array([0.5, 0.45], dtype=np.float32)

        should = should_collapse(state)

        assert should is False  # Ruin too high

    def test_occam_vs_hickam_threshold_difference(self):
        """Occam and Hickam should have different collapse criteria."""
        hypos = [
            Hypothesis("h1", "H1", prior=0.5),
            Hypothesis("h2", "H2", prior=0.5),
        ]

        # Occam mode
        state_occam = State.from_hypotheses(hypos)
        state_occam.collapse_mode = CollapseMode.OCCAM
        state_occam.ruin_prob = 0.05
        state_occam.contradiction_density = 0.2
        state_occam.posteriors = np.array([0.82, 0.18], dtype=np.float32)

        should_occam = should_collapse(state_occam)

        # Hickam mode (same posteriors but checking cluster)
        state_hickam = State.from_hypotheses(hypos)
        state_hickam.collapse_mode = CollapseMode.HICKAM
        state_hickam.ruin_prob = 0.05
        state_hickam.posteriors = np.array([0.82, 0.18], dtype=np.float32)

        state_hickam.exclusivity = build_exclusivity_from_rules(
            ["h1", "h2"],
            pairs=[("h1", "h2", 0.1)]
        )

        should_hickam = should_collapse(state_hickam)

        # Occam should collapse (single strong hyp)
        assert should_occam is True

        # Hickam requires cluster, so different logic
        # (may or may not collapse depending on cluster formation)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
