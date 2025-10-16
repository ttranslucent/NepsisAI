"""Basic tests for NepsisAI."""

import pytest
import numpy as np

from nepsis.core.types import State, Signal, Hypothesis
from nepsis.core.kernel import step, reason
from nepsis.utils.math import softmax, normalize, entropy


class TestMathUtils:
    """Test mathematical utilities."""

    def test_softmax(self):
        """Test softmax function."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = softmax(x)

        assert np.allclose(result.sum(), 1.0)
        assert result[2] > result[1] > result[0]

    def test_normalize(self):
        """Test normalization."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = normalize(x)

        assert np.allclose(result.sum(), 1.0)

    def test_entropy(self):
        """Test entropy calculation."""
        # Uniform distribution has high entropy
        p_uniform = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        h_uniform = entropy(p_uniform)

        # Concentrated distribution has low entropy
        p_concentrated = np.array([0.9, 0.05, 0.03, 0.02], dtype=np.float32)
        h_concentrated = entropy(p_concentrated)

        assert h_uniform > h_concentrated


class TestState:
    """Test State class."""

    def test_from_hypotheses(self):
        """Test state initialization from hypotheses."""
        hypotheses = [
            Hypothesis(id="h1", name="A", prior=0.3),
            Hypothesis(id="h2", name="B", prior=0.7),
        ]

        state = State.from_hypotheses(hypotheses)

        assert len(state.hypotheses) == 2
        assert np.allclose(state.priors.sum(), 1.0)
        assert np.allclose(state.posteriors, state.priors)

    def test_get_top_hypothesis(self):
        """Test getting top hypothesis."""
        hypotheses = [
            Hypothesis(id="h1", name="A", prior=0.2),
            Hypothesis(id="h2", name="B", prior=0.5),
            Hypothesis(id="h3", name="C", prior=0.3),
        ]

        state = State.from_hypotheses(hypotheses)

        # Manually set posteriors
        state.posteriors = np.array([0.1, 0.7, 0.2], dtype=np.float32)

        top = state.get_top_hypothesis(1)[0]
        assert top.name == "B"

    def test_get_posterior(self):
        """Test getting posterior for specific hypothesis."""
        hypotheses = [
            Hypothesis(id="h1", name="A", prior=0.5),
            Hypothesis(id="h2", name="B", prior=0.5),
        ]

        state = State.from_hypotheses(hypotheses)
        state.posteriors = np.array([0.3, 0.7], dtype=np.float32)

        assert np.isclose(state.get_posterior("h1"), 0.3)
        assert np.isclose(state.get_posterior("h2"), 0.7)


class TestReasoning:
    """Test reasoning functions."""

    def test_step(self):
        """Test single reasoning step."""
        hypotheses = [
            Hypothesis(id="h1", name="A", prior=0.5),
            Hypothesis(id="h2", name="B", prior=0.5),
        ]

        state = State.from_hypotheses(hypotheses)
        signal = Signal(type="test", name="s1", value=5.0)

        new_state, ruin_prob = step(state, signal)

        assert new_state.step_num == 1
        assert len(new_state.posterior_history) == 1
        assert 0.0 <= ruin_prob <= 1.0

    def test_reason(self):
        """Test full reasoning process."""
        hypotheses = [
            Hypothesis(id="h1", name="A", prior=0.3),
            Hypothesis(id="h2", name="B", prior=0.4),
            Hypothesis(id="h3", name="C", prior=0.3),
        ]

        signals = [
            Signal(type="test", name="s1", value=3.0),
            Signal(type="test", name="s2", value=5.0),
            Signal(type="test", name="s3", value=2.0),
        ]

        result = reason(signals, hypotheses)

        assert result.top_hypothesis is not None
        assert 0.0 <= result.top_posterior <= 1.0
        assert 0.0 <= result.contradiction_density <= 1.0
        assert result.steps > 0

    def test_red_channel(self):
        """Test red channel pre-emption."""
        hypotheses = [
            Hypothesis(id="h1", name="Safe", prior=0.9),
            Hypothesis(id="h2", name="Danger", prior=0.1),
        ]

        signals = [
            Signal(type="normal", name="s1", value=5.0),
            Signal(type="critical", name="s2", value=50.0, red_threshold=30.0),
        ]

        result = reason(signals, hypotheses)

        assert result.red_preempted is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
