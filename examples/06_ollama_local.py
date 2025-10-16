#!/usr/bin/env python3
"""Example: Using Nepsis with Ollama for local LLM governance.

Requirements:
  1. Install Ollama: https://ollama.com/
  2. Start Ollama: ollama serve
  3. Pull a model: ollama pull llama2

Usage:
  python3 examples/06_ollama_local.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nepsis.stream import govern_completion_stream, Hypothesis


def main():
    """Run governed completion with Ollama."""
    print("=" * 70)
    print("NepsisAI + Ollama: Local LLM Governance")
    print("=" * 70)
    print()

    # Define task
    prompt = """Plan a 3-day trip to Kyoto under $150/day.
Requirements:
- No flights (overland travel only)
- Visit at least 3 cultural sites per day
- Stay within budget"""

    # Define hypotheses
    hypotheses = [
        Hypothesis("budget_ok", "Stay under $150/day budget", prior=0.5),
        Hypothesis("no_flights", "No flights used", prior=0.8),
        Hypothesis("site_count", "3+ sites per day", prior=0.6),
    ]

    print(f"Prompt: {prompt[:60]}...")
    print(f"Model: llama2 (via Ollama)")
    print(f"Hypotheses: {len(hypotheses)}")
    print()
    print("-" * 70)
    print()

    # Stream with Nepsis governance
    stream = govern_completion_stream(
        prompt=prompt,
        model="llama2",  # Will use OllamaAdapter
        hypotheses=hypotheses,
        dynamic_constraints=True,  # Auto-enable trip planner constraint maps
        mode="monitor",
        chunking="hybrid",
        max_tokens_buffer=100,
        emit_every_k_tokens=5,
        base_url="http://localhost:11434"  # Ollama default
    )

    # Process stream
    token_count = 0
    metric_count = 0

    try:
        for event in stream:
            if event.type == "token":
                # Print token
                print(event.payload["text"], end="", flush=True)
                token_count += 1

            elif event.type == "metric":
                # Print metrics
                metric_count += 1
                ρ = event.payload["contradiction_density"]
                fidelity = event.payload["fidelity_score"]
                constraint_risk = event.payload.get("constraint_risk", {})

                print(f"\n\n[Metric {metric_count}]")
                print(f"  Contradiction density (ρ): {ρ:.3f}")
                print(f"  Fidelity score: {fidelity:.3f}")

                # Show constraint risks
                if constraint_risk:
                    print("  Constraint risk:")
                    for hyp_id, risk in constraint_risk.items():
                        status = "⚠️" if risk > 0.5 else "✓"
                        print(f"    {status} {hyp_id}: {risk:.3f}")

                # Warning on high contradiction
                if ρ > 0.5:
                    print(f"  ⚠️  HIGH CONTRADICTION: ρ={ρ:.2f}")

                print()

    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print()
    print("-" * 70)
    print(f"Summary:")
    print(f"  Tokens: {token_count}")
    print(f"  Metrics: {metric_count}")
    print(f"  Provider: Ollama (local)")
    print("=" * 70)


if __name__ == "__main__":
    main()
