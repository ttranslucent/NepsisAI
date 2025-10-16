"""Main streaming API for NepsisAI v0.3.0.

Provides govern_completion_stream() - the primary entry point for
LLM governance middleware.
"""

from __future__ import annotations
from typing import Iterator, List, Optional
from dataclasses import dataclass

from nepsis.core.types import Hypothesis
from nepsis.control.lyapunov import LyapunovWeights
from nepsis.stream.stream_adapter import get_adapter, StreamToken
from nepsis.stream.token_buffer import TokenBuffer, TokenChunk
from nepsis.stream.async_governor import StreamGovernor, GovernorMetrics
from nepsis.stream.metrics_output import StreamEvent, StreamMode
from nepsis.stream.constraint_map import ConstraintMap


@dataclass
class StreamConfig:
    """Configuration for governed streaming."""
    mode: StreamMode = StreamMode.MONITOR
    chunking_mode: str = "hybrid"
    max_tokens_buffer: int = 100
    emit_every_k_tokens: int = 5
    emit_metrics: bool = True

    @classmethod
    def from_dict(cls, config: dict) -> "StreamConfig":
        """Create from dictionary."""
        mode = StreamMode(config.get("mode", "monitor"))
        return cls(
            mode=mode,
            chunking_mode=config.get("chunking", "hybrid"),
            max_tokens_buffer=config.get("max_buffer_tokens", 100),
            emit_every_k_tokens=config.get("emit_every_k_tokens", 5),
            emit_metrics=config.get("emit_metrics", True)
        )


def govern_completion_stream(
    prompt: str,
    model: str = "gpt-4o-mini",
    hypotheses: Optional[List[Hypothesis]] = None,
    constraint_maps: Optional[List[ConstraintMap]] = None,
    dynamic_constraints: bool = False,
    mode: str = "monitor",
    chunking: str = "hybrid",
    max_tokens_buffer: int = 100,
    emit_every_k_tokens: int = 5,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    lyapunov_weights: Optional[LyapunovWeights] = None,
    **llm_kwargs
) -> Iterator[StreamEvent]:
    """Stream LLM completion with Nepsis governance.

    Main API for v0.3.0 streaming middleware.

    Args:
        prompt: Input prompt for LLM
        model: LLM model identifier (e.g., 'gpt-4o-mini', 'claude-3-sonnet', 'llama2')
        hypotheses: Optional list of hypotheses to track
        constraint_maps: Optional constraint maps for drift detection
        dynamic_constraints: Auto-enable default constraint maps
        mode: Governance mode ('monitor', 'guided', 'governed')
        chunking: Chunking strategy ('hybrid', 'semantic', 'fixed')
        max_tokens_buffer: Max tokens before forced chunk emission
        emit_every_k_tokens: Emit metrics every K tokens
        api_key: Optional API key for LLM provider (OpenAI/Anthropic)
        base_url: Optional base URL for Ollama (default: http://localhost:11434)
        lyapunov_weights: Optional Lyapunov weights
        **llm_kwargs: Additional parameters for LLM API

    Yields:
        StreamEvent objects (type='token' or type='metric')

    Example:
        >>> from nepsis.stream import govern_completion_stream, Hypothesis
        >>>
        >>> stream = govern_completion_stream(
        ...     prompt="Plan 3 days in Kyoto under $150/day",
        ...     model="gpt-4o-mini",
        ...     hypotheses=[
        ...         Hypothesis("budget_ok", "Stay under budget", prior=0.5)
        ...     ],
        ...     mode="monitor"
        ... )
        >>>
        >>> for event in stream:
        ...     if event.type == "token":
        ...         print(event.payload["text"], end="")
        ...     elif event.type == "metric":
        ...         print(f"\\n[ρ={event.payload['contradiction_density']:.2f}]")
    """
    # Initialize configuration
    config = StreamConfig(
        mode=StreamMode(mode),
        chunking_mode=chunking,
        max_tokens_buffer=max_tokens_buffer,
        emit_every_k_tokens=emit_every_k_tokens
    )

    # Default hypotheses if not provided
    if hypotheses is None:
        hypotheses = [
            Hypothesis(
                id="coherent",
                name="Output is coherent",
                prior=0.8
            )
        ]

    # Auto-build constraint maps if dynamic_constraints=True
    if dynamic_constraints and constraint_maps is None:
        from nepsis.stream.constraint_map import default_trip_planner_maps
        constraint_maps = default_trip_planner_maps()

    # Initialize components
    adapter = get_adapter(model, api_key=api_key, base_url=base_url)
    buffer = TokenBuffer(
        mode=config.chunking_mode,
        max_buffer_tokens=config.max_tokens_buffer
    )
    governor = StreamGovernor(
        hypotheses=hypotheses,
        lyapunov_weights=lyapunov_weights,
        constraint_maps=constraint_maps  # Pass to governor
    )

    # Stream and govern
    token_count = 0
    chunk_count = 0

    for stream_token in adapter.stream_completion(prompt, model, **llm_kwargs):
        # Emit token event
        yield StreamEvent.token_event(
            text=stream_token.text,
            index=stream_token.index,
            finish_reason=stream_token.finish_reason
        )

        token_count += 1

        # Buffer token and check for chunk
        chunk = buffer.add_token(stream_token.text)

        if chunk:
            # Process chunk with governor
            metrics = governor.process_chunk(chunk)
            chunk_count += 1

            # Emit metric event if configured
            if config.emit_metrics and (chunk_count % config.emit_every_k_tokens == 0):
                yield StreamEvent.metric_event(
                    token_index=metrics.token_index,
                    fidelity_score=metrics.fidelity_score,
                    contradiction_density=metrics.contradiction_density,
                    interpretant_coherence=metrics.interpretant_coherence,
                    constraint_risk=metrics.constraint_risk,
                    collapse_suggest=metrics.collapse_suggest,
                    lyapunov_value=metrics.lyapunov_value,
                    ruin_prob=metrics.ruin_prob
                )

            # Future: Check for intervention in guided/governed modes
            if config.mode == StreamMode.GUIDED:
                if metrics.contradiction_density > 0.70:
                    # TODO: Inject corrective prompt
                    pass

            elif config.mode == StreamMode.GOVERNED:
                if metrics.contradiction_density > 0.90:
                    # TODO: Halt generation
                    pass

    # Final chunk flush
    final_chunk = buffer.flush()
    if final_chunk:
        metrics = governor.process_chunk(final_chunk)

        if config.emit_metrics:
            yield StreamEvent.metric_event(
                token_index=metrics.token_index,
                fidelity_score=metrics.fidelity_score,
                contradiction_density=metrics.contradiction_density,
                interpretant_coherence=metrics.interpretant_coherence,
                constraint_risk=metrics.constraint_risk,
                collapse_suggest=metrics.collapse_suggest,
                lyapunov_value=metrics.lyapunov_value,
                ruin_prob=metrics.ruin_prob
            )
