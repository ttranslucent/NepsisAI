# NepsisAI: Medical-Grade Reasoning Architecture

> A non-ergodic reasoning framework built to medical-grade standards for decision-making under irreversible risk.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/yourusername/nepsisai/releases)

## What This Is

**NepsisAI** is a research architecture for reasoning in domains where:
- Mistakes can be irreversible (non-ergodic)
- Time pressure is extreme
- Uncertainty is inherent
- Safety must pre-empt certainty

Built by an emergency physician with 20 years of pattern-recognition experience.

## What This Is NOT

⚠️ **NOT a clinical decision support tool**
⚠️ **NOT validated for medical use**
⚠️ **NOT a replacement for clinical judgment**

This is a **research prototype** demonstrating architectural principles.

## Quick Start

```bash
# Install core
pip install -e .

# Install with streaming support (v0.3.0+)
pip install -e ".[stream]"

# Run a simple example
python examples/01_simple_reasoning.py

# Run streaming governance demo
python examples/05_streaming_constraint.py

# Run tests
pytest tests/
```

## Core Concepts

- **Triadic Reasoning**: Signal × Interpretant → Evidence → Coherence
- **Non-Ergodic**: Ruin probability never decreases
- **Exclusivity Matrix (Ξ)**: Explicit mutual exclusivity between hypotheses
- **Hickam Multi-Cause**: Multiple compatible diagnoses can co-occur
- **Red-Channel Pre-emption**: Safety before certainty
- **Lyapunov Convergence**: Provable reasoning stability
- **ZeroBack Reset**: Recoverable from epistemic traps

## Architecture Overview

```
Input Signals
    ↓
Red Channel Check → [Pre-empt if danger]
    ↓
Blue Channel:
    1. Interpretant Activation (S→I)
    2. Evidence Modulation (S⊙I)
    3. Coherence Scoring (I-O consistency)
    4. Bayesian Update (π posterior)
    5. Contradiction Check (ρ)
    6. Lyapunov Monitor (V)
    7. Collapse Decision
    ↓
Output: Decision + Audit Trail
```

## Streaming Middleware (v0.3.0 NEW!)

**NepsisAI now wraps LLMs as a meta-governor**, monitoring token streams in real-time for semantic fidelity and constraint adherence.

### Quick Example

```python
from nepsis.stream import govern_completion_stream, Hypothesis

# Wrap any LLM with Nepsis governance
stream = govern_completion_stream(
    prompt="Plan 3 days in Kyoto under $150/day, no flights",
    model="gpt-4o-mini",  # or claude-3-sonnet, etc.
    hypotheses=[
        Hypothesis("budget_ok", "Stay under budget", prior=0.5),
        Hypothesis("no_flights", "No flights used", prior=0.8),
    ],
    mode="monitor"  # monitor | guided | governed
)

for event in stream:
    if event.type == "token":
        print(event.payload["text"], end="")
    elif event.type == "metric":
        # Real-time contradiction density ρ
        print(f"\n[ρ={event.payload['contradiction_density']:.2f}]")
```

### Governance Modes

- **`monitor`** (default): Observe only, emit metrics alongside tokens
- **`guided`** (future): Soft intervention via prompt injection when ρ > threshold
- **`governed`** (future): Hard intervention (halt/restart on Red signals)

### How It Works

1. **Token Buffering**: Hybrid chunking (semantic boundaries OR max buffer)
2. **Real-time Processing**: Each chunk → Nepsis reasoning step
3. **Live Metrics**: Contradiction density ρ, coherence ψ, Lyapunov V
4. **Intervention**: (future) Inject corrective prompts or halt generation

### Supported LLMs

- **OpenAI**: GPT-4, GPT-4o, GPT-3.5
- **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
- **Ollama**: Llama 2/3, Mistral, CodeLlama (local, no API costs!)
- **Coming Soon**: vLLM, Together AI

### Installation

```bash
# Core only
pip install nepsisai

# With streaming support
pip install nepsisai[stream]

# Enterprise (multi-agent coordination)
pip install nepsisai[enterprise]
```

### Using Ollama (Local, Free)

Nepsis supports **local LLMs via Ollama** - no API keys, no costs, full privacy!

```bash
# 1. Install Ollama: https://ollama.com/
# 2. Start server
ollama serve

# 3. Pull a model
ollama pull llama2

# 4. Run with Nepsis
python3 examples/06_ollama_local.py

# 5. Benchmark latency
python3 benchmarks/streaming_latency_ollama.py
```

**Code example:**
```python
from nepsis.stream import govern_completion_stream, Hypothesis

stream = govern_completion_stream(
    prompt="Plan 3 days in Kyoto under $150/day, no flights",
    model="llama2",  # or mistral, codellama, etc.
    hypotheses=[Hypothesis("budget_ok", "Stay under budget", prior=0.5)],
    mode="monitor",
    base_url="http://localhost:11434"  # Ollama default
)

for event in stream:
    if event.type == "token":
        print(event.payload["text"], end="")
```

## Building Exclusivity Matrix

The exclusivity matrix Ξ encodes mutual exclusivity between hypotheses (0.0 = compatible, 1.0 = mutually exclusive):

```python
from nepsis.core import build_exclusivity_from_rules, Hypothesis

# Method 1: Explicit expert rules
excl = build_exclusivity_from_rules(
    ids=["stemi", "angina", "gerd"],
    pairs=[
        ("stemi", "gerd", 0.9),    # Very exclusive
        ("stemi", "angina", 0.6),  # Moderately exclusive
    ],
    groups=[
        (["stemi", "unstable_angina"], 0.8)  # All pairs get 0.8
    ]
)

# Method 2: Infer from hypothesis expectations
from nepsis.core import infer_exclusivity_from_expectations

hypos = {
    "h1": Hypothesis("h1", "STEMI", prior=0.1, expects={"st_elevation": True}),
    "h2": Hypothesis("h2", "GERD", prior=0.3, expects={"st_elevation": False}),
}
excl = infer_exclusivity_from_expectations(hypos)

# Method 3: Migration from v0.1.x dict format
from nepsis.core import exclusivity_from_pairdict

old_format = {("stemi", "gerd"): 0.9}
excl = exclusivity_from_pairdict(hypos, old_format)

# Use in State
state.exclusivity = excl
```

## Migration from v0.1.x → v0.2.0

**⚠️ Breaking Changes:**

### 1. Exclusivity Matrix Required
**Before (v0.1.x):**
```python
# Auto-generated from priors (statistically meaningless!)
state = State.from_hypotheses(hypos)
# Exclusivity was inferred from prior differences
```

**After (v0.2.0):**
```python
from nepsis.core import build_exclusivity_from_rules

# Explicit expert rules required
state = State.from_hypotheses(hypos)
state.exclusivity = build_exclusivity_from_rules(
    ids=["stemi", "gerd", "angina"],
    pairs=[("stemi", "gerd", 0.9), ("stemi", "angina", 0.6)]
)

# Or use inference fallback
from nepsis.core import infer_exclusivity_from_expectations
state.exclusivity = infer_exclusivity_from_expectations(hypos)
```

### 2. Hickam Multi-Cause Collapse
**New in v0.2.0:**
```python
from nepsis.core.types import CollapseMode

# Enable Hickam mode for multi-cause scenarios
state.collapse_mode = CollapseMode.HICKAM

# Greedy clustering algorithm will find compatible hypotheses
# that can co-occur (e.g., CHF + Pneumonia)
```

### 3. Hypothesis.expects Field
**New in v0.2.0:**
```python
# Hypotheses can now specify expected signal values
hypo = Hypothesis(
    id="stemi",
    name="STEMI",
    prior=0.1,
    expects={"st_elevation": True, "troponin_high": True}
)
```

### 4. State Fields
**New fields:**
- `state.exclusivity: Exclusivity` - The Ξ matrix
- `state._index: Dict[str, int]` - O(1) ID→index lookup
- `state.ruin_prob: float` - Monotone ruin probability
- `state.metadata: Dict[str, Any]` - Extensible metadata

**Migration Helper:**
```python
# If you have old dict-based exclusivity
from nepsis.core import exclusivity_from_pairdict

old_excl = {("stemi", "gerd"): 0.9, ("stemi", "angina"): 0.6}
state.exclusivity = exclusivity_from_pairdict(hypos, old_excl)
```

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [Mathematical Theory](docs/THEORY.md)
- [Usage Examples](docs/EXAMPLES.md)
- [Development Roadmap](docs/ROADMAP.md)

## Citation

```bibtex
@software{nepsisai2025,
  author = {Thorn, Trenten Don},
  title = {NepsisAI: Medical-Grade Reasoning Architecture},
  year = {2025},
  url = {https://github.com/yourusername/nepsisai}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Disclaimer

This software is provided for research purposes only. It has not been validated for clinical use and should not be used for medical decisions. Any clinical applications require appropriate regulatory approval, validation studies, and oversight by qualified healthcare professionals.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
