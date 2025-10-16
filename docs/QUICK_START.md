# NepsisAI Quick Start Guide

## Installation

```bash
cd nepsisai
pip install -e .
```

## Your First Reasoning Task

```python
from nepsis import reason, Signal, Hypothesis

# 1. Define what you're reasoning about
hypotheses = [
    Hypothesis(id="h1", name="Viral infection", prior=0.4),
    Hypothesis(id="h2", name="Bacterial infection", prior=0.35),
    Hypothesis(id="h3", name="Fungal infection", prior=0.25),
]

# 2. Provide observations
signals = [
    Signal(type="symptom", name="fever", value=38.5),
    Signal(type="lab", name="wbc_count", value=15.0),
    Signal(type="lab", name="procalcitonin", value=3.2),
]

# 3. Run reasoning
result = reason(signals, hypotheses)

# 4. Get decision
print(f"Top diagnosis: {result.top_hypothesis.name}")
print(f"Confidence: {result.top_posterior:.1%}")
print(f"Converged: {result.converged}")
```

## Key Concepts

### Signal × Interpretant → Hypothesis

Unlike standard AI (Signal → Pattern), NepsisAI uses triadic reasoning:

```
Raw Signal ──┐
             ├──> Modulated Evidence ──> Hypothesis Update
Interpretant ┘
```

The **interpretant** is the "lens" through which signals are interpreted. It encodes context, domain knowledge, and the current reasoning stance.

### Red vs Blue Channels

**Blue Channel**: Normal reasoning flow
- Interpretant modulation
- Bayesian updates
- Coherence checking

**Red Channel**: Safety pre-emption
- Bypasses normal reasoning
- Immediate action on critical signals
- Monotone ruin tracking

### Convergence Metrics

**Contradiction Density (ρ)**: How much you believe mutually exclusive things
- Low ρ = coherent beliefs
- High ρ = confused state → may trigger ZeroBack reset

**Lyapunov Value (V)**: Overall reasoning stability
- Decreasing V = converging
- Stable V = converged

### Decision Modes

**Occam**: Single best hypothesis (low contradiction)
**Hickam**: Multiple hypotheses (complex domains)
**ZeroBack**: Reset to priors (stuck in contradiction)

## Examples

### Basic Reasoning
```bash
python examples/01_simple_reasoning.py
```

Shows complete reasoning flow with audit trail.

### Red Channel Safety
```bash
python examples/02_red_channel.py
```

Demonstrates safety-critical signal pre-emption.

### Iterative Updates
```bash
python examples/03_iterative_reasoning.py
```

Step-by-step reasoning showing state evolution.

## Architecture at a Glance

```
┌─────────────────────────────────────────┐
│           Input Signals                 │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│     Red Channel Check                   │
│     ├─ Pre-empt if safety critical      │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│     Blue Channel Processing             │
│     ├─ 1. Interpretant Activation       │
│     ├─ 2. Signal Modulation             │
│     ├─ 3. Coherence Scoring             │
│     ├─ 4. Bayesian Update               │
│     └─ 5. Contradiction Check (ρ)       │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│     Lyapunov Monitor                    │
│     └─ Check Convergence (V)            │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│     Collapse Decision                   │
│     ├─ Occam: Single hypothesis         │
│     ├─ Hickam: Multiple hypotheses      │
│     └─ ZeroBack: Epistemic reset        │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│     Output + Audit Trail                │
└─────────────────────────────────────────┘
```

## File Reference

**Core reasoning**: `nepsis/core/kernel.py`
- `reason()` - main entry point
- `step()` - single iteration

**Data types**: `nepsis/core/types.py`
- `Signal`, `Hypothesis`, `State`

**Triadic logic**: `nepsis/core/interpretant.py`
- Signal × Interpretant modulation

**Safety**: `nepsis/channels/red.py`
- Red channel pre-emption

**Control**: `nepsis/control/lyapunov.py`
- Convergence tracking

## Common Patterns

### With Audit Trail
```python
from nepsis.utils.logging import AuditTrail

audit = AuditTrail()
result = reason(signals, hypotheses, audit=audit)

# View reasoning steps
print(audit.summary())
```

### Custom Strategy
```python
from nepsis.strategies import Strategy

strategy = Strategy.emergency_medicine()
result = reason(signals, hypotheses,
                lyapunov_weights=strategy.config.get_lyapunov_weights())
```

### Iterative Processing
```python
from nepsis.core.kernel import step
from nepsis.core.types import State

state = State.from_hypotheses(hypotheses)
ruin_prob = 0.0

for signal in signals:
    state, ruin_prob = step(state, signal, ruin_prob=ruin_prob)
    print(f"ρ={state.contradiction_density:.3f}, V={state.lyapunov_value:.3f}")
```

## Next Steps

1. Run the examples
2. Read `ARCHITECTURE.md` for design details
3. Read `THEORY.md` for math foundations
4. Experiment with your own hypotheses and signals
5. Check `ROADMAP.md` for future features

## Disclaimer

This is a **research prototype**. It is NOT validated for clinical use or production deployment. Use for research and experimentation only.
