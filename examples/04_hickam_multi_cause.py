#!/usr/bin/env python3
"""Hickam multi-cause collapse example: CHF + Pneumonia co-occurrence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from nepsis.core.types import State, Hypothesis, Signal, CollapseMode
from nepsis.core.exclusivity_builder import build_exclusivity_from_rules
from nepsis.core.kernel import step
from nepsis.utils.logging import AuditTrail


def main():
    """Demonstrate Hickam collapse with compatible multi-cause scenario."""
    print("=" * 70)
    print("NepsisAI Hickam Multi-Cause Example: CHF + Pneumonia")
    print("=" * 70)
    print()

    # Clinical scenario: Patient can have BOTH CHF and pneumonia
    # (Hickam's dictum: "A patient can have as many diseases as they damn well please")

    hypotheses = [
        Hypothesis(
            id="pneumonia",
            name="Pneumonia",
            prior=0.35,
            expects={"fever": True, "infiltrate": True, "productive_cough": True}
        ),
        Hypothesis(
            id="chf",
            name="CHF (Congestive Heart Failure)",
            prior=0.30,
            expects={"orthopnea": True, "bnp_high": True, "rales": True}
        ),
        Hypothesis(
            id="gerd",
            name="GERD",
            prior=0.20,
            expects={"chest_discomfort": True, "infiltrate": False, "bnp_high": False}
        ),
        Hypothesis(
            id="pe",
            name="Pulmonary Embolism",
            prior=0.15,
            expects={"dyspnea": True, "d_dimer_high": True, "infiltrate": False}
        ),
    ]

    # Initialize state in Hickam mode
    state = State.from_hypotheses(hypotheses)
    state.collapse_mode = CollapseMode.HICKAM

    # Define exclusivity matrix
    # Key insight: CHF and Pneumonia can co-occur (low exclusivity)
    # But most other pairs are incompatible
    state.exclusivity = build_exclusivity_from_rules(
        ids=["pneumonia", "chf", "gerd", "pe"],
        pairs=[
            # Compatible pair (can co-occur)
            ("pneumonia", "chf", 0.2),  # LOW exclusivity

            # Incompatible pairs
            ("pneumonia", "gerd", 0.8),
            ("pneumonia", "pe", 0.7),
            ("chf", "gerd", 0.85),
            ("chf", "pe", 0.6),
            ("gerd", "pe", 0.9),
        ]
    )

    print("Hypotheses:")
    for h in hypotheses:
        print(f"  - {h.name}: prior={h.prior:.2f}")
    print()

    print("Exclusivity Matrix:")
    print(f"  Pneumonia ↔ CHF: {state.exclusivity.get('pneumonia', 'chf'):.2f} (compatible!)")
    print(f"  Pneumonia ↔ GERD: {state.exclusivity.get('pneumonia', 'gerd'):.2f}")
    print(f"  CHF ↔ GERD: {state.exclusivity.get('chf', 'gerd'):.2f}")
    print()

    # Signals suggesting BOTH pneumonia and CHF
    signals = [
        # Pneumonia signals
        Signal(type="symptom", name="fever", value=1.0),
        Signal(type="imaging", name="infiltrate", value=1.0),
        Signal(type="symptom", name="productive_cough", value=1.0),

        # CHF signals
        Signal(type="symptom", name="orthopnea", value=1.0),
        Signal(type="lab", name="bnp_high", value=1.0),
        Signal(type="exam", name="rales", value=1.0),

        # Shared findings
        Signal(type="symptom", name="dyspnea", value=1.0),
    ]

    print("Processing signals...")
    print("-" * 70)

    audit = AuditTrail()

    for i, signal in enumerate(signals, 1):
        print(f"\nStep {i}: {signal.type}:{signal.name} = {signal.value}")

        # Process step
        state, _ = step(state, signal, audit=audit)

        # Show current posteriors
        print(f"  ρ (contradiction): {state.contradiction_density:.3f}")

        print("  Current posteriors:")
        for j, h in enumerate(hypotheses):
            posterior = state.posteriors[j]
            bar = "█" * int(posterior * 40)
            print(f"    {h.name:30s} {posterior:.3f} {bar}")

    print()
    print("=" * 70)
    print("Final Results")
    print("=" * 70)
    print()

    # Check if Hickam cluster formed
    from nepsis.control.collapse import _hickam_cluster_ok, should_collapse

    ok, cluster, mass = _hickam_cluster_ok(state)

    print(f"Hickam Cluster Valid: {ok}")
    if ok:
        print(f"Cluster Members:")
        for h in cluster:
            idx = state._index[h.id]
            print(f"  - {h.name}: posterior={state.posteriors[idx]:.3f}")
        print(f"Total Cluster Mass: {mass:.3f}")
    print()

    print(f"Should Collapse: {should_collapse(state)}")
    print(f"Contradiction Density: {state.contradiction_density:.3f}")
    print(f"Ruin Probability: {state.ruin_prob:.3f}")
    print()

    # Show final distribution
    print("Final Posterior Distribution:")
    for i, h in enumerate(hypotheses):
        posterior = state.posteriors[i]
        bar = "█" * int(posterior * 50)
        print(f"  {h.name:30s} {posterior:.3f} {bar}")
    print()

    print("=" * 70)
    print("Interpretation")
    print("=" * 70)
    print()

    if ok and len(cluster) >= 2:
        print("✓ Hickam's dictum applies: Patient has MULTIPLE diagnoses")
        print(f"✓ Cluster: {', '.join([h.name for h in cluster])}")
        print("✓ Low mutual exclusivity allowed co-occurrence")
    else:
        print("✗ Single diagnosis more likely (Occam's razor)")

    print()
    print("This demonstrates NepsisAI's ability to handle multi-etiology scenarios")
    print("common in complex medical cases, where Occam's razor is insufficient.")


if __name__ == "__main__":
    main()
