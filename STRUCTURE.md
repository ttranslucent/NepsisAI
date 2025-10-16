# NepsisAI v0.1.0 - Structure Overview

## Directory Structure

```
nepsisai/
├── README.md                   # Main project documentation
├── LICENSE                     # MIT License
├── pyproject.toml              # Python project configuration
├── .gitignore                  # Git ignore patterns
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE.md         # System architecture details
│   ├── THEORY.md               # Mathematical foundations
│   ├── EXAMPLES.md             # Usage examples
│   └── ROADMAP.md              # Development roadmap
│
├── nepsis/                     # Main package
│   ├── __init__.py             # Package exports
│   │
│   ├── core/                   # Core reasoning components
│   │   ├── __init__.py
│   │   ├── types.py            # Data structures (State, Signal, Hypothesis)
│   │   ├── kernel.py           # Main reasoning loop (step, reason)
│   │   ├── interpretant.py     # Triadic reasoning (S×I→O)
│   │   └── contradiction.py    # Contradiction density (ρ)
│   │
│   ├── control/                # Control systems
│   │   ├── __init__.py
│   │   ├── lyapunov.py         # Convergence tracking (V)
│   │   └── collapse.py         # Decision modes (Occam/Hickam/ZeroBack)
│   │
│   ├── channels/               # Signal processing
│   │   ├── __init__.py
│   │   ├── red.py              # Safety-critical pre-emption
│   │   └── blue.py             # Normal interpretant reasoning
│   │
│   ├── strategies/             # Domain configurations
│   │   ├── __init__.py
│   │   └── base.py             # Strategy patterns
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── math.py             # Mathematical functions
│       └── logging.py          # Audit trails
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_basic.py           # Unit tests
│   └── fixtures/               # Test data
│
├── examples/                   # Runnable examples
│   ├── 01_simple_reasoning.py  # Basic usage
│   ├── 02_red_channel.py       # Safety pre-emption
│   └── 03_iterative_reasoning.py  # Step-by-step
│
└── scripts/                    # Utility scripts
    └── run_examples.sh         # Run all examples
```

## Core Components

### 1. Types (`nepsis/core/types.py`)
- `Signal`: Observable evidence
- `Hypothesis`: Beliefs about world state
- `State`: Complete reasoning state
- `Evidence`: Processed evidence
- `ReasoningResult`: Final output

### 2. Kernel (`nepsis/core/kernel.py`)
- `step()`: Single reasoning iteration
- `reason()`: Complete reasoning process

### 3. Interpretant Layer (`nepsis/core/interpretant.py`)
- `apply_interpretant()`: S×I modulation
- `coherence_score()`: I-O compatibility
- `triadic_consistency()`: S-I-O alignment

### 4. Control Systems
- **Lyapunov** (`control/lyapunov.py`): Convergence tracking
- **Collapse** (`control/collapse.py`): Decision modes

### 5. Channels
- **Red** (`channels/red.py`): Safety pre-emption
- **Blue** (`channels/blue.py`): Normal reasoning

## Key Algorithms

### Triadic Update
```
S_modulated = S ⊙ (I_states @ Γ_I)
coh = (I_states @ C_IO) ⊙ softmax(L)
π_post ∝ π_prior * L * coh^λ
ρ = Σ Ξ[h1,h2] · (π_post[h1] · π_post[h2])
V = α_c·ρ + α_e·H(π) + α_h·|coh-π| + α_v·||Δπ||
```

### Information Flow
```
Input → Red Check → Blue Process → Lyapunov Monitor → Collapse Decision → Output
```

## Quick Start

### Installation
```bash
cd nepsisai
pip install -e .
```

### Run Tests
```bash
pytest tests/
```

### Run Examples
```bash
# Individual examples
python examples/01_simple_reasoning.py
python examples/02_red_channel.py
python examples/03_iterative_reasoning.py

# All examples
./scripts/run_examples.sh
```

### Basic Usage
```python
from nepsis import reason, Signal, Hypothesis

hypotheses = [
    Hypothesis(id="h1", name="Option A", prior=0.5),
    Hypothesis(id="h2", name="Option B", prior=0.5),
]

signals = [
    Signal(type="observation", name="data1", value=5.0),
    Signal(type="observation", name="data2", value=8.0),
]

result = reason(signals, hypotheses)
print(f"Decision: {result.top_hypothesis.name}")
print(f"Confidence: {result.top_posterior:.3f}")
```

## Design Principles

1. **Pure Functions**: No side effects, deterministic
2. **Explicit State**: All state visible and mutable
3. **Auditable**: Every step logged
4. **Modular**: Components independently testable
5. **Type-Safe**: Clear interfaces and contracts

## Next Steps

1. Run examples to see the system in action
2. Read `docs/ARCHITECTURE.md` for design details
3. Read `docs/THEORY.md` for mathematical background
4. Explore `docs/EXAMPLES.md` for usage patterns
5. Check `docs/ROADMAP.md` for future development

## Status

**Version**: 0.1.0 (Alpha)
**Status**: Research prototype
**Warning**: NOT validated for production use

This is a clean-slate implementation of the NepsisAI architecture, ready for development and experimentation.
