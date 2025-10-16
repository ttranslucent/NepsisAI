"""Logging and audit trail utilities."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import sys


def setup_logger(name: str = "nepsis", level: int = logging.INFO) -> logging.Logger:
    """Setup a logger with standard formatting.

    Args:
        name: Logger name
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


@dataclass
class StepRecord:
    """Record of a single reasoning step."""
    step_num: int
    signal_type: str
    signal_name: str
    signal_value: float
    contradiction_density: float
    lyapunov_value: float
    top_hypothesis: str
    top_posterior: float
    red_preempted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditTrail:
    """Complete audit trail of reasoning process."""
    steps: List[StepRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, record: StepRecord) -> None:
        """Add a step record to the trail."""
        self.steps.append(record)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "steps": [
                {
                    "step": s.step_num,
                    "signal": f"{s.signal_type}:{s.signal_name}={s.signal_value}",
                    "ρ": s.contradiction_density,
                    "V": s.lyapunov_value,
                    "top": f"{s.top_hypothesis} ({s.top_posterior:.3f})",
                    "red": s.red_preempted,
                }
                for s in self.steps
            ],
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = ["Reasoning Audit Trail", "=" * 50]

        for step in self.steps:
            lines.append(
                f"Step {step.step_num}: {step.signal_type}:{step.signal_name}={step.signal_value:.2f}"
            )
            lines.append(f"  → ρ={step.contradiction_density:.3f}, V={step.lyapunov_value:.3f}")
            lines.append(f"  → Top: {step.top_hypothesis} ({step.top_posterior:.3f})")
            if step.red_preempted:
                lines.append("  → RED CHANNEL PREEMPTED")
            lines.append("")

        return "\n".join(lines)
