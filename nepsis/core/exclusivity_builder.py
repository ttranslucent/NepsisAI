"""Exclusivity matrix builders for hypothesis mutual exclusivity.

Provides tools to build exclusivity matrices from:
1. Expert rules (explicit pairs/groups)
2. Inferred from hypothesis expectations (heuristic)
3. Migration from old dict format
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from nepsis.core.types import Hypothesis


class Exclusivity:
    """Typed exclusivity matrix with hypothesis ordering.

    Represents mutual exclusivity Ξ[i,j] ∈ [0,1] between hypotheses.
    - 0.0 = compatible (can co-occur)
    - 1.0 = mutually exclusive

    Properties:
    - Symmetric: Ξ[i,j] = Ξ[j,i]
    - Zero diagonal: Ξ[i,i] = 0
    """

    def __init__(self, ids: List[str], matrix: NDArray[np.float32], default: float = 0.0):
        """Initialize exclusivity matrix.

        Args:
            ids: Ordered list of hypothesis IDs
            matrix: Square symmetric matrix, shape (len(ids), len(ids))
            default: Default exclusivity for unknown pairs
        """
        self.ids = ids
        self.matrix = matrix
        self.default = default
        self._index = {h: i for i, h in enumerate(ids)}

        # Validate
        assert matrix.shape == (len(ids), len(ids)), "Matrix shape mismatch"
        assert np.allclose(matrix, matrix.T), "Matrix must be symmetric"
        assert np.allclose(np.diag(matrix), 0.0), "Diagonal must be zero"

    def get(self, h1: str, h2: str) -> float:
        """Get exclusivity between two hypotheses.

        This is the primary public API for exclusivity lookup.

        Args:
            h1: First hypothesis ID
            h2: Second hypothesis ID

        Returns:
            Exclusivity value ∈ [0,1] where:
            - 0.0 = compatible (can co-occur)
            - 1.0 = mutually exclusive
            - Returns `self.default` if either ID is unknown

        Properties:
            - Symmetric: get(a, b) == get(b, a)
            - Thread-safe O(1) lookup
            - Diagonal: get(a, a) == 0.0

        Example:
            >>> excl = build_exclusivity_from_rules(
            ...     ids=["stemi", "gerd"],
            ...     pairs=[("stemi", "gerd", 0.9)]
            ... )
            >>> excl.get("stemi", "gerd")
            0.9
            >>> excl.get("gerd", "stemi")  # Symmetric
            0.9
            >>> excl.get("unknown", "stemi")  # Unknown ID
            0.0
        """
        if h1 not in self._index or h2 not in self._index:
            return self.default
        i, j = self._index[h1], self._index[h2]
        return float(self.matrix[i, j])

    def __repr__(self) -> str:
        return f"Exclusivity(n={len(self.ids)}, nonzero={np.count_nonzero(self.matrix)//2})"


def build_exclusivity_from_rules(
    ids: List[str],
    pairs: Optional[Iterable[Tuple[str, str, float]]] = None,
    groups: Optional[Iterable[Tuple[List[str], float]]] = None,
    default: float = 0.0,
) -> Exclusivity:
    """Build exclusivity matrix from expert rules.

    Args:
        ids: Ordered list of hypothesis IDs
        pairs: List of (h1, h2, exclusivity) tuples
        groups: List of (hypothesis_list, exclusivity) tuples
                All pairs within group get same exclusivity
        default: Default exclusivity for unspecified pairs

    Returns:
        Exclusivity matrix

    Example:
        >>> excl = build_exclusivity_from_rules(
        ...     ids=["stemi", "angina", "gerd"],
        ...     pairs=[
        ...         ("stemi", "gerd", 0.9),
        ...         ("stemi", "angina", 0.6)
        ...     ],
        ...     groups=[
        ...         (["stemi", "angina"], 0.6)  # redundant but allowed
        ...     ]
        ... )
    """
    n = len(ids)
    idx = {h: i for i, h in enumerate(ids)}
    M = np.zeros((n, n), dtype=np.float32)

    # Apply pair rules
    if pairs:
        for a, b, xi in pairs:
            if a in idx and b in idx:
                i, j = idx[a], idx[b]
                xi_clamped = max(0.0, min(1.0, float(xi)))
                M[i, j] = M[j, i] = xi_clamped

    # Apply group rules
    if groups:
        for group, xi in groups:
            xi_clamped = max(0.0, min(1.0, float(xi)))
            for i_a, a in enumerate(group):
                for b in group[i_a + 1:]:
                    if a in idx and b in idx:
                        ia, ib = idx[a], idx[b]
                        # Take max if multiple rules specify same pair
                        M[ia, ib] = M[ib, ia] = max(M[ia, ib], xi_clamped)

    # Ensure diagonal is zero
    np.fill_diagonal(M, 0.0)

    return Exclusivity(ids=ids, matrix=M, default=default)


def infer_exclusivity_from_expectations(
    hypos: Dict[str, Hypothesis],
    signal_weights: Optional[Dict[str, float]] = None,
    default: float = 0.0,
) -> Exclusivity:
    """Infer exclusivity heuristically from hypothesis expectations.

    Heuristic: Exclusivity ∝ weighted fraction of conflicting expectations

    Args:
        hypos: Hypothesis dictionary
        signal_weights: Optional weights for each signal type
        default: Default exclusivity

    Returns:
        Exclusivity matrix

    Example:
        >>> h1 = Hypothesis("h1", prior=0.5, expects={"fever": True, "cough": True})
        >>> h2 = Hypothesis("h2", prior=0.5, expects={"fever": False, "cough": True})
        >>> excl = infer_exclusivity_from_expectations({"h1": h1, "h2": h2})
        >>> # fever conflicts → exclusivity = 0.5 (1 of 2 signals)
    """
    ids = list(hypos.keys())
    n = len(ids)
    idx = {h: i for i, h in enumerate(ids)}
    M = np.zeros((n, n), dtype=np.float32)
    sw = signal_weights or {}

    for i in range(n):
        for j in range(i + 1, n):
            hi, hj = hypos[ids[i]], hypos[ids[j]]

            # Get overlapping expectation keys
            common = set(hi.expects.keys()).intersection(hj.expects.keys())
            if not common:
                continue

            # Count weighted conflicts
            conflict, weight_sum = 0.0, 0.0
            for k in common:
                w = float(sw.get(k, 1.0))
                weight_sum += w
                if hi.expects[k] != hj.expects[k]:
                    conflict += w

            # Exclusivity = fraction of conflicting expectations
            xi = (conflict / weight_sum) if weight_sum > 0 else 0.0
            M[i, j] = M[j, i] = xi

    np.fill_diagonal(M, 0.0)
    return Exclusivity(ids=ids, matrix=M, default=default)


def exclusivity_from_pairdict(
    hypos: Dict[str, Hypothesis],
    pairdict: Dict[Tuple[str, str], float],
    default: float = 0.0,
) -> Exclusivity:
    """Migration adapter: Convert old dict format to new Exclusivity.

    Args:
        hypos: Hypothesis dictionary
        pairdict: Old format: {("h1", "h2"): 0.9, ...}
        default: Default exclusivity

    Returns:
        Exclusivity matrix

    Example:
        >>> # Old code
        >>> state.exclusivity = {("stemi", "gerd"): 0.9}
        >>>
        >>> # Migration
        >>> state.exclusivity = exclusivity_from_pairdict(
        ...     hypos=state.hypos,
        ...     pairdict={("stemi", "gerd"): 0.9}
        ... )
    """
    ids = list(hypos.keys())
    pairs = [(h1, h2, xi) for (h1, h2), xi in pairdict.items()]
    return build_exclusivity_from_rules(ids=ids, pairs=pairs, default=default)
