from typing import List, Dict, Tuple
import numpy as np
import pandas as pd


def recommend_top_k(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    title: str,
    k: int = 10,
) -> List[Tuple[str, float]]:
    """
    Recommend top-k most similar movies.
    
    Args:
        df: DataFrame with movie data (must have 'title' and 'title_with_year' columns)
        sim_matrix: Similarity matrix (N x N)
        title: Movie title to find recommendations for
        k: Number of recommendations to return
    
    Returns:
        List of tuples: (title_with_year, similarity_score)
    """
    title_to_idx = {t: i for i, t in enumerate(df["title"].tolist())}
    if title not in title_to_idx:
        raise ValueError(f"Title '{title}' not found.")
    idx = title_to_idx[title]
    sims = sim_matrix[idx].copy()  # Copy to avoid mutating the original matrix
    sims[idx] = -1.0
    
    if k < len(sims):
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
    else:
        top_idx = np.argsort(sims)[::-1][:k]
    
    # Use title_with_year for display if available, otherwise fall back to title
    if "title_with_year" in df.columns:
        return [(df.iloc[i]["title_with_year"], float(sims[i])) for i in top_idx]
    else:
        return [(df.iloc[i]["title"], float(sims[i])) for i in top_idx]
