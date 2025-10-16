# Usage Examples

## Basic Reasoning

```python
from nepsis import reason, Signal, Hypothesis

# Define hypotheses
hypotheses = [
    Hypothesis(id="h1", name="Sepsis", prior=0.1),
    Hypothesis(id="h2", name="Pneumonia", prior=0.3),
    Hypothesis(id="h3", name="CHF", prior=0.2),
]

# Observe signals
signals = [
    Signal(type="vital", name="fever", value=39.5),
    Signal(type="lab", name="wbc", value=18000),
    Signal(type="symptom", name="dyspnea", value=1.0),
]

# Reason
result = reason(signals, hypotheses)

print(f"Top hypothesis: {result.top_hypothesis}")
print(f"Contradiction density: {result.contradiction_density:.3f}")
print(f"Converged: {result.converged}")
```

## Iterative Reasoning

```python
from nepsis import State, step

# Initialize state
state = State.from_hypotheses(hypotheses)

# Process signals iteratively
for signal in signals:
    state = step(state, signal)
    print(f"After {signal.name}: ρ={state.contradiction_density:.3f}")

# Check convergence
if state.lyapunov_stable:
    print(f"Converged to: {state.get_top_hypothesis()}")
else:
    print("Reasoning unstable - consider ZeroBack reset")
```

## Custom Strategy

```python
from nepsis.strategies import StrategyConfig

# Emergency medicine strategy
ed_strategy = StrategyConfig(
    name="emergency_medicine",
    collapse_mode="hickam",  # Multiple diagnoses expected
    red_channel_threshold=0.8,  # High safety sensitivity
    lyapunov_weights={
        "contradiction": 2.0,
        "entropy": 1.0,
        "coherence": 1.5,
        "velocity": 0.5,
    }
)

result = reason(signals, hypotheses, strategy=ed_strategy)
```

## Red Channel Pre-emption

```python
from nepsis.channels import check_red_preempt

# Critical signal triggers immediate action
critical_signal = Signal(
    type="vital",
    name="sbp",
    value=60,  # Shock
    red_threshold=80
)

if check_red_preempt(critical_signal):
    # Safety action pre-empts normal reasoning
    print("RED CHANNEL: Immediate intervention required")
else:
    # Continue normal blue-channel reasoning
    state = step(state, critical_signal)
```

## Audit Trail

```python
from nepsis.utils import AuditTrail

# Enable detailed logging
audit = AuditTrail()

result = reason(signals, hypotheses, audit=audit)

# Examine reasoning steps
for step_record in audit.steps:
    print(f"Step {step_record.step_num}:")
    print(f"  Signal: {step_record.signal}")
    print(f"  ρ: {step_record.contradiction_density}")
    print(f"  V: {step_record.lyapunov_value}")
    print(f"  Decision: {step_record.decision}")
```

See `examples/` directory for runnable scripts.
