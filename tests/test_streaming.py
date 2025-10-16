"""Tests for streaming components (v0.3.0)."""

import pytest
from nepsis.stream.token_buffer import TokenBuffer, TokenChunk, chunk_tokens
from nepsis.stream.metrics_output import StreamEvent, StreamMode, MetricsFormatter


class TestTokenBuffer:
    """Test token buffering and chunking."""

    def test_hybrid_mode_sentence_boundary(self):
        """Test hybrid mode emits on sentence boundaries."""
        buffer = TokenBuffer(mode="hybrid", max_buffer_tokens=100)

        tokens = ["The", " quick", " brown", " fox", "."]
        chunks = []

        for token in tokens:
            chunk = buffer.add_token(token)
            if chunk:
                chunks.append(chunk)

        # Should emit one chunk after period
        assert len(chunks) == 1
        assert chunks[0].text == "The quick brown fox."
        assert chunks[0].is_complete is True

    def test_hybrid_mode_max_buffer_fallback(self):
        """Test hybrid mode falls back to max buffer."""
        buffer = TokenBuffer(mode="hybrid", max_buffer_tokens=5)

        # Long sequence without sentence boundary
        tokens = ["word"] * 10
        chunks = []

        for token in tokens:
            chunk = buffer.add_token(token)
            if chunk:
                chunks.append(chunk)

        # Should emit at least one chunk due to max buffer
        assert len(chunks) >= 1
        assert chunks[0].token_count == 5
        assert chunks[0].is_complete is False  # Not semantic boundary

    def test_fixed_mode(self):
        """Test fixed mode emits every N tokens."""
        buffer = TokenBuffer(mode="fixed", max_buffer_tokens=3)

        tokens = ["a", "b", "c", "d", "e", "f"]
        chunks = []

        for token in tokens:
            chunk = buffer.add_token(token)
            if chunk:
                chunks.append(chunk)

        # Should emit 2 chunks of 3 tokens each
        assert len(chunks) == 2
        assert all(c.token_count == 3 for c in chunks)

    def test_flush_remaining(self):
        """Test flushing remaining buffer."""
        buffer = TokenBuffer(mode="hybrid", max_buffer_tokens=100)

        tokens = ["incomplete", " sentence"]
        for token in tokens:
            buffer.add_token(token)

        # No chunks yet (no sentence boundary)
        final = buffer.flush()

        assert final is not None
        assert final.text == "incomplete sentence"
        assert final.is_complete is True  # Flush marks as complete

    def test_chunk_tokens_iterator(self):
        """Test convenience iterator function."""
        tokens = ["Hello", " world", ".", " How", " are", " you", "?"]

        chunks = list(chunk_tokens(iter(tokens), mode="hybrid"))

        # Should get 2 chunks (two sentences)
        assert len(chunks) == 2
        assert chunks[0].text == "Hello world."
        assert chunks[1].text == " How are you?"


class TestStreamEvent:
    """Test stream event formatting."""

    def test_token_event(self):
        """Test token event creation."""
        event = StreamEvent.token_event(text="hello", index=42)

        assert event.type == "token"
        assert event.payload["text"] == "hello"
        assert event.payload["index"] == 42

    def test_metric_event(self):
        """Test metric event creation."""
        event = StreamEvent.metric_event(
            token_index=100,
            fidelity_score=0.94,
            contradiction_density=0.06,
            interpretant_coherence=0.88,
            constraint_risk={"budget_ok": 0.12},
            collapse_suggest="none"
        )

        assert event.type == "metric"
        assert event.payload["tkn_idx"] == 100
        assert event.payload["fidelity_score"] == 0.94
        assert event.payload["contradiction_density"] == 0.06

    def test_metrics_formatter_inline(self):
        """Test inline metrics formatting."""
        event = StreamEvent.metric_event(
            token_index=100,
            fidelity_score=0.94,
            contradiction_density=0.06,
            interpretant_coherence=0.88,
            constraint_risk={},
            collapse_suggest="none",
            lyapunov_value=1.2
        )

        inline = MetricsFormatter.format_inline(event)
        assert "ρ=0.06" in inline
        assert "V=1.2" in inline
        assert "ψ=0.88" in inline

    def test_metrics_formatter_table(self):
        """Test table row formatting."""
        event = StreamEvent.metric_event(
            token_index=100,
            fidelity_score=0.94,
            contradiction_density=0.06,
            interpretant_coherence=0.88,
            constraint_risk={},
            collapse_suggest="none"
        )

        row = MetricsFormatter.format_table_row(event)
        assert "100" in row
        assert "0.94" in row
        assert "0.06" in row
        assert "none" in row


class TestStreamMode:
    """Test stream mode enum."""

    def test_mode_values(self):
        """Test mode enum values."""
        assert StreamMode.MONITOR.value == "monitor"
        assert StreamMode.GUIDED.value == "guided"
        assert StreamMode.GOVERNED.value == "governed"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        mode = StreamMode("monitor")
        assert mode == StreamMode.MONITOR


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
