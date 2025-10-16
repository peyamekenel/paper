import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
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


def zscore_normalize(scores: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Apply z-score normalization (standardization) to scores.
    
    More robust than min-max when scores are tightly clustered.
    If std is too small or non-finite, returns zeros.
    
    Args:
        scores: Array of scores to normalize
        epsilon: Small value to prevent division by zero
    
    Returns:
        Z-score normalized scores (mean=0, std=1), or zeros if degenerate
    """
    scores = np.asarray(scores, dtype=float)
    
    if len(scores) == 0:
        return scores
    
    mean = np.nanmean(scores)
    std = np.nanstd(scores)
    
    if not np.isfinite(mean) or not np.isfinite(std) or std < epsilon:
        return np.zeros_like(scores)
    
    return (scores - mean) / (std + epsilon)


def hybrid_fusion_zscore(
    semantic_scores: np.ndarray, 
    tfidf_scores: np.ndarray, 
    alpha: float = 0.5
) -> np.ndarray:
    """
    Robust hybrid fusion using z-score normalization.
    
    Normalizes each signal with z-scores before blending. More robust
    than per-query min-max when scores are tightly clustered.
    
    Args:
        semantic_scores: Semantic similarity scores
        tfidf_scores: TF-IDF similarity scores
        alpha: Weight for semantic scores (0-1), TF-IDF gets (1-alpha)
    
    Returns:
        Hybrid scores combining both signals
    """
    if len(semantic_scores) != len(tfidf_scores):
        raise ValueError("Score arrays must have same length")
    
    if len(semantic_scores) == 0:
        return np.array([])
    
    stack = np.vstack([semantic_scores, tfidf_scores]).T
    scaler = StandardScaler(with_mean=True, with_std=True)
    
    try:
        zs = scaler.fit_transform(stack)
        hybrid_scores = alpha * zs[:, 0] + (1 - alpha) * zs[:, 1]
        return hybrid_scores
    except Exception:
        return np.zeros(len(semantic_scores))
