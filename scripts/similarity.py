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
    
    Args:
        scores: Array of scores to normalize
        epsilon: Small value to prevent division by zero
    
    Returns:
        Normalized scores in [0, 1] range
    """
    min_score = scores.min()
    max_score = scores.max()
    score_range = max_score - min_score
    
    if score_range < epsilon:
        return np.ones_like(scores)
    
    return (scores - min_score) / (score_range + epsilon)
