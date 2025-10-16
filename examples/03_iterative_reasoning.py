#!/usr/bin/env python3
"""Iterative reasoning example showing step-by-step updates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nepsis.core.types import State, Signal, Hypothesis
from nepsis.core.kernel import step
from nepsis.utils.logging import AuditTrail


def main():
    """Run iterative reasoning example."""
    print("=" * 70)
    print("NepsisAI Iterative Reasoning Example")
    print("=" * 70)
    print()

    # Hypotheses
    hypotheses = [
        Hypothesis(id="h1", name="Viral", prior=0.40),
        Hypothesis(id="h2", name="Bacterial", prior=0.35),
        Hypothesis(id="h3", name="Fungal", prior=0.15),
    ]

    # Initialize state
    state = State.from_hypotheses(hypotheses)
    ruin_prob = 0.0

    # Signals to process iteratively
    signals = [
        Signal(type="symptom", name="fever", value=38.5),
        Signal(type="lab", name="wbc", value=12.0),
        Signal(type="lab", name="procalcitonin", value=2.5),
        Signal(type="culture", name="gram_stain", value=8.0),
    ]

    audit = AuditTrail()

    print("Processing signals step by step...")
    print()

    for i, signal in enumerate(signals):
        print(f"Step {i+1}: {signal.type}:{signal.name} = {signal.value}")
        print("-" * 70)

        # Process one step
        state, ruin_prob = step(state, signal, audit=audit, ruin_prob=ruin_prob)

        # Show current state
        print(f"  Contradiction density (ρ): {state.contradiction_density:.3f}")
        print(f"  Lyapunov value (V): {state.lyapunov_value:.3f}")
        print(f"  Converged: {state.lyapunov_stable}")
        print()

        print("  Current posteriors:")
        for j, h in enumerate(hypotheses):
            posterior = state.posteriors[j]
            bar = "█" * int(posterior * 40)
            print(f"    {h.name:12s} {posterior:.3f} {bar}")
        print()

        if state.lyapunov_stable:
            print("  ✓ Reasoning converged!")
            break

        print()

    print("=" * 70)
    print(f"Final decision: {state.get_top_hypothesis()[0].name}")
    print(f"Confidence: {state.posteriors.max():.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
