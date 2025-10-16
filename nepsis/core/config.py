"""Global configuration parameters for NepsisAI reasoning.

Domain-specific strategies can override these at runtime.
"""

# Red channel & ruin
RED_RUIN_THRESHOLD = 0.50  # Ruin probability threshold for red escalation
COLLAPSE_MAX_RUIN = 0.10   # Max ruin allowed for collapse

# Collapse thresholds
COLLAPSE_MIN_TOP = 0.80     # Min posterior for Occam collapse
COLLAPSE_MAX_CONTRA = 0.25  # Max contradiction for Occam collapse

# Hickam multi-cause collapse
HICKAM_MASS_THRESH = 0.85   # Min total mass for Hickam cluster
HICKAM_MAX_PAIR_XI = 0.40   # Max pairwise exclusivity within cluster

# Triadic reasoning
LAMBDA_LIKE = 0.40  # Likelihood strength in evidence modulation
COH_POWER = 0.50    # Coherence exponent λ

# ZeroBack rollback
ZERO_BACK_CONTRA = 0.70     # Contradiction threshold for ZeroBack
ZERO_BACK_STALL_STEPS = 4   # Steps to detect stall

# STILL checkpoints
STILL_PERIOD = 3  # Checkpoint every N steps

# Lyapunov stability
LYAPUNOV_WEIGHTS = {
    "contradiction": 2.0,
    "entropy": 1.0,
    "coherence": 1.5,
    "velocity": 0.5,
}
