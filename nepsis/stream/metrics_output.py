"""Metrics output formatting for stream events.

Defines event types and JSON schema for v0.3.0 streaming API.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Literal
from enum import Enum


class StreamMode(Enum):
    """Stream governance modes."""
    MONITOR = "monitor"  # Observe only, emit metrics
    GUIDED = "guided"    # Soft intervention via prompt injection
    GOVERNED = "governed"  # Hard intervention (halt/restart) - future


@dataclass
class StreamEvent:
    """Unified stream event for tokens and metrics.

    Event Types:
        - token: LLM output token
        - metric: Nepsis governance metric
        - intervention: Guidance or halt signal (future)
    """
    type: Literal["token", "metric", "intervention"]
    payload: Dict[str, Any]

    @classmethod
    def token_event(cls, text: str, index: int, **metadata) -> "StreamEvent":
        """Create token event.

        Args:
            text: Token text
            index: Token index
            **metadata: Additional metadata

        Returns:
            StreamEvent with type='token'
        """
        return cls(
            type="token",
            payload={
                "text": text,
                "index": index,
                **metadata
            }
        )

    @classmethod
    def metric_event(
        cls,
        token_index: int,
        fidelity_score: float,
        contradiction_density: float,
        interpretant_coherence: float,
        constraint_risk: Dict[str, float],
        collapse_suggest: str,
        lyapunov_value: float = 0.0,
        ruin_prob: float = 0.0,
        **extra
    ) -> "StreamEvent":
        """Create metric event (v0.3.0 schema).

        Args:
            token_index: Current token index
            fidelity_score: Semantic fidelity (1 - ρ)
            contradiction_density: Contradiction metric ρ
            interpretant_coherence: Mean coherence score
            constraint_risk: Per-hypothesis posterior
            collapse_suggest: Collapse mode suggestion
            lyapunov_value: Lyapunov stability metric
            ruin_prob: Non-ergodic ruin probability
            **extra: Additional metrics

        Returns:
            StreamEvent with type='metric'
        """
        return cls(
            type="metric",
            payload={
                "type": "metric",
                "tkn_idx": token_index,
                "fidelity_score": round(fidelity_score, 3),
                "contradiction_density": round(contradiction_density, 3),
                "interpretant_coherence": round(interpretant_coherence, 3),
                "constraint_risk": {
                    k: round(v, 3) for k, v in constraint_risk.items()
                },
                "collapse_suggest": collapse_suggest,
                "lyapunov_value": round(lyapunov_value, 3),
                "ruin_prob": round(ruin_prob, 3),
                **extra
            }
        )

    @classmethod
    def intervention_event(
        cls,
        action: Literal["inject_prompt", "halt", "restart"],
        reason: str,
        **metadata
    ) -> "StreamEvent":
        """Create intervention event (future: guided/governed modes).

        Args:
            action: Intervention action
            reason: Human-readable reason
            **metadata: Additional context

        Returns:
            StreamEvent with type='intervention'
        """
        return cls(
            type="intervention",
            payload={
                "action": action,
                "reason": reason,
                **metadata
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict())


class MetricsFormatter:
    """Formats metrics for different output modes."""

    @staticmethod
    def format_inline(event: StreamEvent) -> str:
        """Format metric as inline annotation.

        Args:
            event: Metric event

        Returns:
            Inline string like " [ρ=0.06 V=1.2]"

        Example:
            >>> event = StreamEvent.metric_event(
            ...     token_index=100,
            ...     fidelity_score=0.94,
            ...     contradiction_density=0.06,
            ...     interpretant_coherence=0.88,
            ...     constraint_risk={"budget_ok": 0.12},
            ...     collapse_suggest="none"
            ... )
            >>> MetricsFormatter.format_inline(event)
            ' [ρ=0.06 V=0.0 ψ=0.88]'
        """
        if event.type != "metric":
            return ""

        p = event.payload
        return (
            f" [ρ={p['contradiction_density']:.2f} "
            f"V={p['lyapunov_value']:.1f} "
            f"ψ={p['interpretant_coherence']:.2f}]"
        )

    @staticmethod
    def format_table_row(event: StreamEvent) -> str:
        """Format metric as table row.

        Args:
            event: Metric event

        Returns:
            Table row string

        Example:
            >>> MetricsFormatter.format_table_row(event)
            '  100 | 0.94 | 0.06 | 0.88 | none'
        """
        if event.type != "metric":
            return ""

        p = event.payload
        return (
            f"{p['tkn_idx']:>5} | "
            f"{p['fidelity_score']:.2f} | "
            f"{p['contradiction_density']:.2f} | "
            f"{p['interpretant_coherence']:.2f} | "
            f"{p['collapse_suggest']:>8}"
        )

    @staticmethod
    def table_header() -> str:
        """Get table header."""
        return "  Tkn | Fid  | ρ    | ψ    | Collapse"
