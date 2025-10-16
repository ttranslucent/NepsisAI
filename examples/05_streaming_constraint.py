#!/usr/bin/env python3
"""Streaming constraint satisfaction with Nepsis governance.

Demonstrates v0.3.0 middleware: LLM streaming with real-time monitoring.

Task: Plan 3-day trip to Kyoto
Constraints:
  - Budget: $150/day maximum
  - No flights allowed
  - Must visit 3+ cultural sites per day

Nepsis tracks constraint violations in real-time via contradiction density ρ.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nepsis.stream import govern_completion_stream, Hypothesis
from nepsis.stream.metrics_output import MetricsFormatter


def main():
    """Run streaming constraint example."""
    print("=" * 70)
    print("NepsisAI v0.3.0: Streaming Constraint Governance")
    print("=" * 70)
    print()

    # Define task
    prompt = """Plan a 3-day trip to Kyoto with the following constraints:
- Budget: Maximum $150 per day (including accommodation, food, transport)
- Transportation: No flights allowed (train/bus only)
- Cultural sites: Visit at least 3 cultural/historical sites per day

Provide a detailed day-by-day itinerary with cost breakdown."""

    # Define hypotheses (constraints to track)
    hypotheses = [
        Hypothesis(
            id="budget_ok",
            name="Stay under $150/day budget",
            prior=0.5
        ),
        Hypothesis(
            id="no_flights",
            name="No flights used",
            prior=0.8
        ),
        Hypothesis(
            id="cultural_sites",
            name="3+ cultural sites per day",
            prior=0.6
        )
    ]

    print("Constraints to Monitor:")
    for h in hypotheses:
        print(f"  - {h.name} (prior={h.prior:.1f})")
    print()

    print("Task:")
    print(prompt)
    print()

    print("-" * 70)
    print("Streaming with Nepsis Governance (monitor mode)")
    print("-" * 70)
    print()

    # Stream with governance
    try:
        # Note: Requires OpenAI API key in environment
        # Set: export OPENAI_API_KEY="..."

        stream = govern_completion_stream(
            prompt=prompt,
            model="gpt-4o-mini",
            hypotheses=hypotheses,
            mode="monitor",
            chunking="hybrid",
            max_tokens_buffer=100,
            emit_every_k_tokens=5,
            temperature=0.7,
            max_tokens=1000
        )

        print(MetricsFormatter.table_header())
        print("-" * 70)

        full_response = []

        for event in stream:
            if event.type == "token":
                # Print token
                text = event.payload["text"]
                print(text, end="", flush=True)
                full_response.append(text)

            elif event.type == "metric":
                # Print metric row
                print()  # New line before metric
                print(MetricsFormatter.format_table_row(event))

                # Check for high contradiction
                ρ = event.payload["contradiction_density"]
                if ρ > 0.70:
                    print(f"  ⚠️  HIGH CONTRADICTION (ρ={ρ:.2f}) - Potential constraint violation!")

        print()
        print()
        print("=" * 70)
        print("Summary")
        print("=" * 70)
        print()
        print(f"Total response length: {len(''.join(full_response))} chars")
        print()

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print()
        print("To run streaming examples, install:")
        print("  pip install nepsisai[stream]")
        print()
        return

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("Make sure to set OPENAI_API_KEY environment variable:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print()
        return

    print("✅ Streaming governance demo complete!")
    print()
    print("Nepsis monitored the LLM output in real-time,")
    print("tracking semantic fidelity and constraint adherence.")
    print()


if __name__ == "__main__":
    main()
