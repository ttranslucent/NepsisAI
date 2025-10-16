# NepsisAI Architecture

## Overview

NepsisAI implements triadic reasoning: `Signal × Interpretant → Evidence → Coherence`

This differs from standard dyadic systems (`Signal → Pattern → Output`) by adding an explicit interpretant layer that modulates how evidence is processed.

## Core Components

### 1. Triadic Tensor Operations (`nepsis/core/interpretant.py`)

The interpretant layer implements:
- S→I gating (signal activates interpretant states)
- I→O compatibility (interpretant-hypothesis alignment)
- Coherence verification (triadic consistency check)

### 2. Contradiction Density (`nepsis/core/contradiction.py`)

Measures mutual exclusivity violations:
```
ρ = Σ Ξ[h1,h2] · (π_post[h1] · π_post[h2])
```

Where Ξ is the contradiction matrix encoding logical conflicts.

### 3. Collapse Governor (`nepsis/core/collapse.py`)

Three decision modes:
- **Occam**: Single hypothesis when ρ low
- **Hickam**: Multiple hypotheses when domain expects it
- **ZeroBack**: Epistemic reset when stuck

### 4. Lyapunov Convergence (`nepsis/control/lyapunov.py`)

Tracks reasoning stability via:
```
V = V_contradiction + V_entropy + V_coherence + V_velocity
```

Guarantees convergence when dV/dt < 0.

### 5. Red/Blue Channels (`nepsis/channels/`)

- **Red**: Safety-critical signals, pre-empt blue reasoning
- **Blue**: Interpretant-driven exploration
- **Ruin**: Monotone probability tracking

## Information Flow

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

## Design Principles

1. **Pure Functions**: All core ops return new state, no side effects
2. **Explicit State**: No hidden global variables
3. **Deterministic**: Same input → same output (for testing)
4. **Auditable**: Every step logged with minimal payload
5. **Monotone Ruin**: Safety never relaxes

## Domain Adaptation

Strategy configs in `nepsis/strategies/` tune:
- Collapse thresholds
- Lyapunov weights
- Red-channel sensitivity
- Interpretant priors

See individual domain docs for details.
