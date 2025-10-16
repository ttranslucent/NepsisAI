#!/usr/bin/env python3
"""Latency benchmarks: Baseline vs Nepsis-governed streaming.

Critical validation for v0.3.0: Measure overhead of Nepsis middleware.

Target: <15% overhead for monitor mode (acceptable for audit use cases)
"""

import sys
import time
from pathlib import Path
from typing import List
import statistics

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nepsis.stream import govern_completion_stream, Hypothesis


def benchmark_baseline(prompt: str, model: str, runs: int = 10) -> List[float]:
    """Benchmark raw LLM streaming (no Nepsis).

    Args:
        prompt: Test prompt
        model: Model identifier
        runs: Number of trials

    Returns:
        List of completion times (seconds)
    """
    try:
        import openai
    except ImportError:
        print("❌ OpenAI package required. Install: pip install openai")
        return []

    times = []

    for i in range(runs):
        start = time.time()

        try:
            response = openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                max_tokens=500
            )

            token_count = 0
            for chunk in response:
                if chunk.choices[0].delta.content:
                    token_count += 1

            elapsed = time.time() - start
            times.append(elapsed)

            print(f"  Run {i+1}/{runs}: {elapsed:.2f}s ({token_count} tokens)")

        except Exception as e:
            print(f"  Run {i+1}/{runs}: ERROR - {e}")
            continue

    return times


def benchmark_nepsis(
    prompt: str,
    model: str,
    hypotheses: List[Hypothesis],
    runs: int = 10
) -> List[float]:
    """Benchmark Nepsis-governed streaming.

    Args:
        prompt: Test prompt
        model: Model identifier
        hypotheses: Hypotheses to track
        runs: Number of trials

    Returns:
        List of completion times (seconds)
    """
    times = []

    for i in range(runs):
        start = time.time()

        try:
            stream = govern_completion_stream(
                prompt=prompt,
                model=model,
                hypotheses=hypotheses,
                mode="monitor",
                chunking="hybrid",
                max_tokens_buffer=100,
                emit_every_k_tokens=5,
                max_tokens=500
            )

            token_count = 0
            metric_count = 0

            for event in stream:
                if event.type == "token":
                    token_count += 1
                elif event.type == "metric":
                    metric_count += 1

            elapsed = time.time() - start
            times.append(elapsed)

            print(f"  Run {i+1}/{runs}: {elapsed:.2f}s ({token_count} tokens, {metric_count} metrics)")

        except Exception as e:
            print(f"  Run {i+1}/{runs}: ERROR - {e}")
            continue

    return times


def print_results(baseline_times: List[float], nepsis_times: List[float]):
    """Print benchmark results table.

    Args:
        baseline_times: Baseline completion times
        nepsis_times: Nepsis completion times
    """
    if not baseline_times or not nepsis_times:
        print("\n❌ Insufficient data for comparison")
        return

    baseline_mean = statistics.mean(baseline_times)
    baseline_std = statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0

    nepsis_mean = statistics.mean(nepsis_times)
    nepsis_std = statistics.stdev(nepsis_times) if len(nepsis_times) > 1 else 0

    overhead_pct = ((nepsis_mean / baseline_mean) - 1) * 100

    print("\n" + "=" * 70)
    print("Latency Benchmark Results")
    print("=" * 70)
    print()
    print(f"{'Metric':<30} {'Baseline':<20} {'Nepsis':<20}")
    print("-" * 70)
    print(f"{'Mean time (s)':<30} {baseline_mean:<20.3f} {nepsis_mean:<20.3f}")
    print(f"{'Std dev (s)':<30} {baseline_std:<20.3f} {nepsis_std:<20.3f}")
    print(f"{'Min time (s)':<30} {min(baseline_times):<20.3f} {min(nepsis_times):<20.3f}")
    print(f"{'Max time (s)':<30} {max(baseline_times):<20.3f} {max(nepsis_times):<20.3f}")
    print()
    print(f"{'Overhead':<30} {'-':<20} {overhead_pct:>19.1f}%")
    print()

    # Verdict
    if overhead_pct < 15:
        print("✅ PASS: Overhead <15% (acceptable for audit use cases)")
    elif overhead_pct < 25:
        print("⚠️  WARNING: Overhead 15-25% (marginal, consider optimization)")
    else:
        print("❌ FAIL: Overhead >25% (must optimize before launch)")
    print()


def main():
    """Run latency benchmarks."""
    print("=" * 70)
    print("NepsisAI v0.3.0: Latency Benchmark")
    print("=" * 70)
    print()

    # Test configuration
    prompt = "Plan a 3-day trip to Kyoto under $150/day, no flights allowed."
    model = "gpt-4o-mini"
    runs = 10

    hypotheses = [
        Hypothesis("budget_ok", "Stay under budget", prior=0.5),
        Hypothesis("no_flights", "No flights used", prior=0.8),
    ]

    print(f"Prompt: {prompt}")
    print(f"Model:  {model}")
    print(f"Runs:   {runs}")
    print(f"Hypotheses: {len(hypotheses)}")
    print()

    # Check for API key
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print()
        print("Set your API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        print()
        return

    # Baseline benchmark
    print("-" * 70)
    print("Baseline (raw OpenAI streaming)")
    print("-" * 70)
    baseline_times = benchmark_baseline(prompt, model, runs)
    print()

    # Nepsis benchmark
    print("-" * 70)
    print("Nepsis-governed streaming")
    print("-" * 70)
    nepsis_times = benchmark_nepsis(prompt, model, hypotheses, runs)
    print()

    # Results
    print_results(baseline_times, nepsis_times)

    # Save results
    import json
    results = {
        "prompt": prompt,
        "model": model,
        "runs": runs,
        "baseline": {
            "times": baseline_times,
            "mean": statistics.mean(baseline_times) if baseline_times else 0,
            "std": statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0
        },
        "nepsis": {
            "times": nepsis_times,
            "mean": statistics.mean(nepsis_times) if nepsis_times else 0,
            "std": statistics.stdev(nepsis_times) if len(nepsis_times) > 1 else 0
        },
        "overhead_pct": ((statistics.mean(nepsis_times) / statistics.mean(baseline_times)) - 1) * 100 if baseline_times and nepsis_times else 0
    }

    with open("benchmarks/latency_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to: benchmarks/latency_results.json")
    print()


if __name__ == "__main__":
    main()
