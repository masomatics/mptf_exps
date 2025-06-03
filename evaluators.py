import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import log

import numpy as np
from typing import Optional

def cover_entropy(points: np.ndarray, eps: float, *, metric: str = "euclidean") -> float:
    """
    Kolmogorov ε-entropy of a finite set of points.

    Parameters
    ----------
    points : ndarray, shape (N, d)
        The particle positions at one time step.
    eps    : float
        Covering radius (same units as `points`).
    metric : str
        Any metric accepted by scikit-learn's NearestNeighbors
        ("euclidean", "cosine", "minkowski", etc.).

    Returns
    -------
    H_eps : float
        log N(ε) where N(ε) is the *approximate* minimum # of ε-balls
        covering the cloud (greedy k-centre algorithm, 1-to-2× optimal in
        practice).
    """
    P = points.copy()
    N = P.shape[0]
    if N == 0:
        return -np.inf  # empty set ⇒ log 0 = -∞ by convention

    # --- greedy k-centre ------------------------------------------------
    covered = np.zeros(N, dtype=bool)
    nn = NearestNeighbors(radius=eps, metric=metric).fit(P)
    centres = 0
    while not covered.all():
        # pick the farthest uncovered point as a new centre
        idx = np.where(~covered)[0][0]
        new_centre = P[idx].reshape(1, -1)

        # mark everything within eps of the new centre as covered
        neigh_idx = nn.radius_neighbors(new_centre, eps, return_distance=False)[0]
        covered[neigh_idx] = True
        centres += 1

    return log(centres)






def proximity_probability(
    pts: np.ndarray,
    delta: float,
    *,
    sample_pairs: Optional[int] = None,
    assume_unit: bool = True,
) -> float:
    """
    Empirical estimate of
    
        P( ⟨x₁, x₂⟩ ≥ 1 − δ ),
    
    where (x₁, x₂) are two independent draws *without replacement*
    from the point cloud ``pts``.

    Parameters
    ----------
    pts : ndarray, shape (N, d)
        The population at time *t*.  Each row is a vector xᵢ(t).
    delta : float
        Threshold parameter in the event ⟨x₁, x₂⟩ ≥ 1−δ.
    sample_pairs : int or None, optional
        • If None (default) and N ≤ 10 000, evaluate **all** N · (N−1)/2 pairs.  
        • Otherwise draw this many random unordered pairs for a Monte-Carlo estimate.
    assume_unit : bool, default True
        Set to False if the points are *not* already ℓ²-normalised; the
        function will normalise them first.

    Returns
    -------
    prob : float
        The empirical probability.
    """
    pts = np.asarray(pts, dtype=float)
    N, d = pts.shape
    if N < 2:
        raise ValueError("Need at least two samples")

    # normalise if requested
    if not assume_unit:
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        # avoid division by zero
        pts = pts / np.where(norms == 0, 1.0, norms)

    # decide whether to enumerate or sample
    full_enum = sample_pairs is None and N <= 10_000
    rng = np.random.default_rng()

    if full_enum:
        # vectorised upper-triangular inner products
        G = pts @ pts.T                               # Gram matrix
        iu = np.triu_indices(N, k=1)
        hits = np.count_nonzero(G[iu] >= 1.0 - delta)
        total = iu[0].size
    else:
        if sample_pairs is None:
            sample_pairs = 100_000                    # default MC budget
        idx1 = rng.integers(0, N, size=sample_pairs)
        idx2 = rng.integers(0, N - 1, size=sample_pairs)
        # ensure idx2 ≠ idx1 by simple trick
        idx2 = np.where(idx2 >= idx1, idx2 + 1, idx2)
        inner_prods = np.einsum("ij,ij->i", pts[idx1], pts[idx2])
        hits = np.count_nonzero(inner_prods >= 1.0 - delta)
        total = sample_pairs

    return hits / total
