# NepsisAI: Medical-Grade Reasoning Architecture

> A non-ergodic reasoning framework built to medical-grade standards for decision-making under irreversible risk.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
# Install
pip install -e .

# Run a simple example
python examples/01_simple_reasoning.py

# Run tests
pytest tests/
```

## Core Concepts

- **Triadic Reasoning**: Signal × Interpretant → Evidence → Coherence
- **Non-Ergodic**: Ruin probability never decreases
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
