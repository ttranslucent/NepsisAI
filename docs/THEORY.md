# Mathematical Foundations

## Triadic Semiotics (Peirce)

Every reasoning step involves:
- **Sign** (S): Observable evidence
- **Object** (O): Hypotheses about reality
- **Interpretant** (I): Meaning-making lens

Standard AI is dyadic: S → O (signal to pattern)

NepsisAI is triadic: S × I → O (signal modulated by interpretant)

## The Core Equations

### 1. Evidence Modulation
```
S_modulated = S ⊙ (I_states @ Γ_I)
```
Raw signals filtered through interpretant context.

### 2. Coherence Score
```
coh = (I_states @ C_IO) ⊙ softmax(L)
```
How well does this hypothesis fit the current interpretive stance?

### 3. Posterior Update
```
π_post ∝ π_prior * L * coh^λ
```
Bayesian update modulated by coherence.

### 4. Contradiction Density
```
ρ = Σ_{h1,h2} Ξ[h1,h2] · (π_post[h1] · π_post[h2])
```
Measures belief in mutually exclusive hypotheses.

### 5. Lyapunov Function
```
V = α_c·ρ + α_e·H(π) + α_h·|coh-π| + α_v·||Δπ||
```
Reasoning energy - must decrease to converge.

## Non-Ergodic Decision Theory

Standard RL: E[reward] over all trajectories

Non-ergodic: P(ruin) along *this* trajectory

Key insight: time-average ≠ ensemble-average when ruin possible.

NepsisAI: ruin_prob is monotone, red-path pre-empts.

## Convergence Guarantees

**Theorem (informal):** If:
1. Coherence function locally convex
2. Contradiction density decreasing with coherence
3. ZeroBack resets prevent local minima
4. Sufficient evidence exists

Then recursive interpretation converges to I* where:
- C(S,O,I*) maximal
- ρ(I*) < threshold
- I* stable under perturbation

(Formal proof in preparation)

## Biological Grounding

Cellular signaling: E + S ⇌ ES → E + P

NepsisAI: Evidence + Hypothesis ⇌ Bound → Interpretation

Allostery: Global coherence modulates all edge weights.

Prevents runaway activation (hallucination in AI, cancer in cells).
