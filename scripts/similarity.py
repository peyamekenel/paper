import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


def cosine_sim_dense(mat: np.ndarray) -> np.ndarray:
    return cosine_similarity(mat)


def cosine_sim_sparse(mat: sparse.spmatrix) -> np.ndarray:
    return cosine_similarity(mat, dense_output=True)
