#!/usr/bin/env python3
"""Red channel example demonstrating safety pre-emption."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nepsis import reason, Signal, Hypothesis
from nepsis.utils.logging import AuditTrail


def main():
    """Run red channel example."""
    print("=" * 70)
    print("NepsisAI Red Channel Pre-emption Example")
    print("=" * 70)
    print()

    # Hypotheses
    hypotheses = [
        Hypothesis(id="h1", name="Stable", prior=0.80),
        Hypothesis(id="h2", name="Unstable", prior=0.15),
        Hypothesis(id="h3", name="Critical", prior=0.05),
    ]

    # Normal signals followed by critical red signal
    signals = [
        Signal(type="vital", name="hr", value=85),
        Signal(type="vital", name="rr", value=16),
        Signal(type="vital", name="sbp", value=120),
        # Critical hypotension - triggers red channel
        Signal(type="vital", name="sbp_drop", value=55, red_threshold=70),
        # These won't matter much after red pre-emption
        Signal(type="lab", name="lactate", value=8.0),
    ]

    audit = AuditTrail()

    print("Signals (note red threshold on signal 4):")
    for i, s in enumerate(signals, 1):
        red_flag = f" [RED THRESHOLD: {s.red_threshold}]" if s.red_threshold else ""
        print(f"  {i}. {s.type}:{s.name} = {s.value}{red_flag}")
    print()

    # Run reasoning
    result = reason(signals, hypotheses, audit=audit)

    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()
    print(f"Red Channel Pre-empted: {result.red_preempted}")
    print(f"Top Hypothesis: {result.top_hypothesis.name}")
    print(f"Posterior: {result.top_posterior:.3f}")
    print()

    # Audit trail shows red pre-emption
    print("=" * 70)
    print("Audit Trail (showing red pre-emption)")
    print("=" * 70)
    print()
    print(audit.summary())


if __name__ == "__main__":
    main()
