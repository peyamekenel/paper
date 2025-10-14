import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


def cosine_sim_dense(mat: np.ndarray) -> np.ndarray:
    return cosine_similarity(mat)


def cosine_sim_sparse(mat: sparse.spmatrix) -> np.ndarray:
    return cosine_similarity(mat, dense_output=True)


def minmax_normalize(scores: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Apply min-max normalization to scale scores to [0, 1] range.
    
    If the score range is degenerate (all values equal or non-finite),
    returns a zero vector to avoid distorting the hybrid ranking.
    
    Args:
        scores: Array of scores to normalize
        epsilon: Small value to prevent division by zero
    
    Returns:
        Normalized scores in [0, 1] range, or zeros if degenerate
    """
    scores = np.asarray(scores, dtype=float)
    mn = np.nanmin(scores)
    mx = np.nanmax(scores)
    rng = mx - mn
    
    if not np.isfinite(mn) or not np.isfinite(mx) or rng < epsilon:
        return np.zeros_like(scores)
    
    return (scores - mn) / (rng + epsilon)
