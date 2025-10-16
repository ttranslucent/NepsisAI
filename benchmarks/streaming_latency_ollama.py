#!/usr/bin/env python3
"""Latency benchmarks using Ollama (local, no API costs).

Run this after starting Ollama:
  ollama serve
  ollama pull llama2  # or your preferred model

Usage:
  python3 benchmarks/streaming_latency_ollama.py
"""

import sys
import time
from pathlib import Path
from typing import List
import statistics

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nepsis.stream import govern_completion_stream, Hypothesis


def check_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running.

    Args:
        base_url: Ollama server URL

    Returns:
        True if available, False otherwise
    """
    try:
        import requests
        response = requests.get(f"{base_url}/api/tags")
        return response.status_code == 200
    except Exception:
        return False


def benchmark_baseline_ollama(
    prompt: str,
    model: str,
    runs: int = 5,
    base_url: str = "http://localhost:11434"
) -> List[float]:
    """Benchmark raw Ollama streaming (no Nepsis).

    Args:
        prompt: Test prompt
        model: Model identifier
        runs: Number of trials
        base_url: Ollama server URL

    Returns:
        List of completion times (seconds)
    """
    try:
        import requests
    except ImportError:
        print("❌ Requests package required. Install: pip install requests")
        return []

    times = []

    for i in range(runs):
        start = time.time()

        try:
            url = f"{base_url}/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
            }

            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()

            token_count = 0
            for line in response.iter_lines():
                if line:
                    import json
                    chunk = json.loads(line)
                    if "response" in chunk and chunk["response"]:
                        token_count += 1

            elapsed = time.time() - start
            times.append(elapsed)

            print(f"  Run {i+1}/{runs}: {elapsed:.2f}s ({token_count} tokens)")

        except Exception as e:
            print(f"  Run {i+1}/{runs}: ERROR - {e}")
            continue

    return times


def benchmark_nepsis_ollama(
    prompt: str,
    model: str,
    hypotheses: List[Hypothesis],
    runs: int = 5,
    base_url: str = "http://localhost:11434"
) -> List[float]:
    """Benchmark Nepsis-governed streaming with Ollama.

    Args:
        prompt: Test prompt
        model: Model identifier
        hypotheses: Hypotheses to track
        runs: Number of trials
        base_url: Ollama server URL

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
                base_url=base_url
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
            import traceback
            traceback.print_exc()
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
    print("Latency Benchmark Results (Ollama)")
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
    """Run latency benchmarks with Ollama."""
    print("=" * 70)
    print("NepsisAI v0.3.0: Latency Benchmark (Ollama)")
    print("=" * 70)
    print()

    # Check Ollama availability
    base_url = "http://localhost:11434"
    if not check_ollama_available(base_url):
        print(f"❌ Error: Ollama server not available at {base_url}")
        print()
        print("Start Ollama:")
        print("  ollama serve")
        print()
        print("Pull a model:")
        print("  ollama pull llama2")
        print("  ollama pull mistral")
        print()
        return

    # Test configuration
    prompt = "Plan a 3-day trip to Kyoto under $150/day, no flights allowed."
    model = "llama2"  # Change to your preferred model
    runs = 5  # Fewer runs for local testing

    hypotheses = [
        Hypothesis("budget_ok", "Stay under budget", prior=0.5),
        Hypothesis("no_flights", "No flights used", prior=0.8),
    ]

    print(f"Prompt: {prompt}")
    print(f"Model:  {model}")
    print(f"Runs:   {runs}")
    print(f"Hypotheses: {len(hypotheses)}")
    print(f"Ollama: {base_url}")
    print()

    # Baseline benchmark
    print("-" * 70)
    print("Baseline (raw Ollama streaming)")
    print("-" * 70)
    baseline_times = benchmark_baseline_ollama(prompt, model, runs, base_url)
    print()

    # Nepsis benchmark
    print("-" * 70)
    print("Nepsis-governed streaming")
    print("-" * 70)
    nepsis_times = benchmark_nepsis_ollama(prompt, model, hypotheses, runs, base_url)
    print()

    # Results
    print_results(baseline_times, nepsis_times)

    # Save results
    import json
    results = {
        "prompt": prompt,
        "model": model,
        "provider": "ollama",
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

    output_file = Path(__file__).parent / "latency_results_ollama.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
