"""Tests for exclusivity matrix building and operations."""

import pytest
import numpy as np

from nepsis.core.types import Hypothesis, State
from nepsis.core.exclusivity_builder import (
    Exclusivity,
    build_exclusivity_from_rules,
    infer_exclusivity_from_expectations,
    exclusivity_from_pairdict,
)
from nepsis.core.contradiction import compute_contradiction_density


class TestExclusivityBuilder:
    """Test exclusivity matrix construction."""

    def test_build_from_rules_pairs(self):
        """Test building exclusivity from explicit pairs."""
        ids = ["h1", "h2", "h3"]
        pairs = [
            ("h1", "h2", 0.9),
            ("h1", "h3", 0.5),
        ]

        excl = build_exclusivity_from_rules(ids, pairs=pairs)

        assert np.isclose(excl.get("h1", "h2"), 0.9)
        assert np.isclose(excl.get("h2", "h1"), 0.9)  # Symmetric
        assert np.isclose(excl.get("h1", "h3"), 0.5)
        assert np.isclose(excl.get("h2", "h3"), 0.0)  # Not specified

    def test_build_from_rules_groups(self):
        """Test building exclusivity from groups."""
        ids = ["h1", "h2", "h3"]
        groups = [
            (["h1", "h2", "h3"], 0.8),  # All mutually exclusive
        ]

        excl = build_exclusivity_from_rules(ids, groups=groups)

        # All pairs should have exclusivity 0.8
        assert np.isclose(excl.get("h1", "h2"), 0.8)
        assert np.isclose(excl.get("h1", "h3"), 0.8)
        assert np.isclose(excl.get("h2", "h3"), 0.8)

    def test_exclusivity_properties(self):
        """Test that exclusivity matrix has required properties."""
        ids = ["a", "b", "c"]
        pairs = [("a", "b", 0.7), ("b", "c", 0.3)]

        excl = build_exclusivity_from_rules(ids, pairs=pairs)

        # Symmetric
        M = excl.matrix
        assert np.allclose(M, M.T)

        # Zero diagonal
        assert np.allclose(np.diag(M), 0.0)

        # Values in [0, 1]
        assert np.all(M >= 0.0)
        assert np.all(M <= 1.0)

    def test_infer_from_expectations(self):
        """Test inferring exclusivity from conflicting expectations."""
        hypos = {
            "h1": Hypothesis("h1", "A", prior=0.5, expects={"fever": True, "cough": True}),
            "h2": Hypothesis("h2", "B", prior=0.5, expects={"fever": False, "cough": True}),
        }

        excl = infer_exclusivity_from_expectations(hypos)

        # Should have some exclusivity because fever conflicts
        # 1 of 2 signals conflicts → exclusivity ≈ 0.5
        xi = excl.get("h1", "h2")
        assert 0.4 < xi < 0.6

    def test_migration_adapter(self):
        """Test backward compatibility adapter."""
        hypos = {
            "stemi": Hypothesis("stemi", "STEMI", prior=0.4),
            "gerd": Hypothesis("gerd", "GERD", prior=0.3),
        }

        old_format = {
            ("stemi", "gerd"): 0.9,
        }

        excl = exclusivity_from_pairdict(hypos, old_format)

        assert np.isclose(excl.get("stemi", "gerd"), 0.9)
        assert np.isclose(excl.get("gerd", "stemi"), 0.9)


class TestContradictionWithExclusivity:
    """Test contradiction density with new exclusivity system."""

    def test_zero_contradiction_no_overlap(self):
        """No contradiction when hypotheses don't conflict."""
        hypos = [
            Hypothesis("h1", "A", prior=0.5),
            Hypothesis("h2", "B", prior=0.5),
        ]

        state = State.from_hypotheses(hypos)

        # No exclusivity set
        state.exclusivity = build_exclusivity_from_rules(
            ["h1", "h2"],
            pairs=[]  # No conflicts
        )

        # Both hypotheses likely
        state.posteriors = np.array([0.6, 0.5], dtype=np.float32)

        rho = compute_contradiction_density(state)

        # No exclusivity → no contradiction
        assert rho == 0.0

    def test_high_contradiction_mutual_exclusion(self):
        """High contradiction when mutually exclusive hypotheses both likely."""
        hypos = [
            Hypothesis("h1", "A", prior=0.5),
            Hypothesis("h2", "B", prior=0.5),
        ]

        state = State.from_hypotheses(hypos)

        # High mutual exclusivity
        state.exclusivity = build_exclusivity_from_rules(
            ["h1", "h2"],
            pairs=[("h1", "h2", 0.9)]
        )

        # Both hypotheses very likely
        state.posteriors = np.array([0.7, 0.6], dtype=np.float32)

        rho = compute_contradiction_density(state)

        # ρ = 0.5 * (p^T Ξ p) where p = [0.7, 0.6]
        # = 0.5 * (0.7 * 0.9 * 0.6 + 0.6 * 0.9 * 0.7)  [symmetric]
        # = 0.5 * 2 * 0.9 * 0.7 * 0.6
        # = 0.9 * 0.7 * 0.6
        expected = 0.9 * 0.7 * 0.6
        assert np.isclose(rho, expected, atol=0.01)

    def test_vectorized_vs_naive(self):
        """Verify vectorized computation matches naive loop."""
        hypos = [
            Hypothesis("h1", "A", prior=0.3),
            Hypothesis("h2", "B", prior=0.4),
            Hypothesis("h3", "C", prior=0.3),
        ]

        state = State.from_hypotheses(hypos)

        state.exclusivity = build_exclusivity_from_rules(
            ["h1", "h2", "h3"],
            pairs=[("h1", "h2", 0.8), ("h2", "h3", 0.6)]
        )

        state.posteriors = np.array([0.4, 0.5, 0.3], dtype=np.float32)

        # Vectorized
        rho_vec = compute_contradiction_density(state)

        # Naive computation
        p = state.posteriors
        Ξ = state.exclusivity.matrix
        rho_naive = 0.0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                rho_naive += Ξ[i, j] * p[i] * p[j]

        assert np.isclose(rho_vec, rho_naive, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
