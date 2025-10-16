"""Streaming middleware for LLM governance.

NepsisAI v0.3.0: Token-level semantic monitoring and intervention.

Main API:
    govern_completion_stream() - Wrap LLM streaming with Nepsis monitoring

Modes:
    - monitor: Observe only, emit metrics (default)
    - guided: Soft intervention via prompt injection
    - governed: Hard intervention (halt/restart) - future

Example:
    >>> from nepsis.stream import govern_completion_stream, Hypothesis
    >>>
    >>> stream = govern_completion_stream(
    ...     prompt="Plan 3 days in Kyoto under $150/day",
    ...     model="gpt-4o-mini",
    ...     hypotheses=[
    ...         Hypothesis("budget_ok", "Stay under budget", prior=0.5),
    ...     ],
    ...     mode="monitor"
    ... )
    >>>
    >>> for event in stream:
    ...     if event.type == "token":
    ...         print(event.text, end="")
    ...     elif event.type == "metric":
    ...         print(f"\\n[ρ={event.payload['contradiction_density']:.2f}]")
"""

from nepsis.stream.api import govern_completion_stream, StreamEvent, StreamMode
from nepsis.core.types import Hypothesis

__all__ = [
    "govern_completion_stream",
    "StreamEvent",
    "StreamMode",
    "Hypothesis",
]
