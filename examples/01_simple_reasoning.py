#!/usr/bin/env python3
"""Simple reasoning example demonstrating NepsisAI basics."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nepsis import reason, Signal, Hypothesis
from nepsis.utils.logging import AuditTrail


def main():
    """Run simple reasoning example."""
    print("=" * 70)
    print("NepsisAI Simple Reasoning Example")
    print("=" * 70)
    print()

    # Define hypotheses (simplified medical scenarios)
    hypotheses = [
        Hypothesis(id="h1", name="Sepsis", prior=0.15),
        Hypothesis(id="h2", name="Pneumonia", prior=0.30),
        Hypothesis(id="h3", name="CHF", prior=0.20),
        Hypothesis(id="h4", name="COPD", prior=0.25),
    ]

    print("Hypotheses:")
    for h in hypotheses:
        print(f"  - {h.name}: prior={h.prior:.2f}")
    print()

    # Define signals (observations)
    signals = [
        Signal(type="vital", name="fever", value=39.5),
        Signal(type="lab", name="wbc", value=18.0),
        Signal(type="symptom", name="dyspnea", value=8.0),
        Signal(type="lab", name="procalcitonin", value=5.2),
    ]

    print("Signals:")
    for s in signals:
        print(f"  - {s.type}:{s.name} = {s.value}")
    print()

    # Create audit trail
    audit = AuditTrail()

    # Run reasoning
    print("Running reasoning...")
    print("-" * 70)

    result = reason(signals, hypotheses, audit=audit)

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()
    print(f"Top Hypothesis: {result.top_hypothesis.name}")
    print(f"Posterior Probability: {result.top_posterior:.3f}")
    print(f"Contradiction Density (ρ): {result.contradiction_density:.3f}")
    print(f"Converged: {result.converged}")
    print(f"Steps: {result.steps}")
    print()

    # Show all posteriors
    print("Final Posteriors:")
    for i, h in enumerate(result.state.hypotheses):
        posterior = result.state.posteriors[i]
        bar = "█" * int(posterior * 50)
        print(f"  {h.name:15s} {posterior:.3f} {bar}")
    print()

    # Show audit trail
    print("=" * 70)
    print("Audit Trail")
    print("=" * 70)
    print()
    print(audit.summary())


if __name__ == "__main__":
    main()
