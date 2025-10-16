"""
Diversity algorithms for recommendation re-ranking.
Implements MMR (Maximal Marginal Relevance) and other diversity methods.
"""
from typing import List, Tuple, Set
import numpy as np
import pandas as pd


def mmr_rerank(
    query_idx: int,
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    similarity_matrix: np.ndarray,
    k: int,
    lambda_param: float = 0.5,
    semantic_similarity_matrix: np.ndarray = None,
    diversity_alpha: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-rank recommendations using Maximal Marginal Relevance (MMR).
    
    MMR balances relevance and diversity by selecting items that are:
    1. Similar to the query (relevance)
    2. Different from already selected items (diversity)
    
    Formula: MMR = argmax[λ × Sim(d, q) - (1-λ) × max(Sim(d, d_i))]
    
    Supports multi-view diversity: combines semantic and TF-IDF similarities
    for more robust diversity penalty.
    
    Args:
        query_idx: Index of the query item
        candidate_indices: Indices of candidate items (sorted by relevance)
        candidate_scores: Relevance scores for candidates
        similarity_matrix: TF-IDF similarity matrix (N x N) for diversity
        k: Number of items to select
        lambda_param: Tradeoff between relevance and diversity
                     - λ=1.0: Pure relevance (no diversity)
                     - λ=0.5: Balanced
                     - λ=0.0: Pure diversity (not recommended)
        semantic_similarity_matrix: Optional semantic similarity matrix for multi-view diversity
        diversity_alpha: Weight for semantic vs TF-IDF in diversity penalty (0-1)
                        Only used if semantic_similarity_matrix is provided
    
    Returns:
        Tuple of (selected_indices, mmr_scores)
    """
    if len(candidate_indices) == 0:
        return np.array([]), np.array([])
    
    k = min(k, len(candidate_indices))
    
    # Initialize
    selected_indices = []
    selected_scores = []
    remaining = set(range(len(candidate_indices)))
    
    for _ in range(k):
        if not remaining:
            break
        
        best_mmr_score = -np.inf
        best_idx = None
        
        for i in remaining:
            candidate_idx = candidate_indices[i]
            relevance_score = candidate_scores[i]
            
            # Compute diversity penalty
            if len(selected_indices) == 0:
                # First item: pure relevance
                diversity_penalty = 0.0
            else:
                # Max similarity to already selected items
                selected_global_indices = [candidate_indices[j] for j in selected_indices]
                
                # TF-IDF diversity penalty
                if hasattr(similarity_matrix, "getrow"):
                    row = similarity_matrix.getrow(candidate_idx)[:, selected_global_indices]
                    tfidf_penalty = float(row.max()) if row.nnz else 0.0
                else:
                    similarities_to_selected = similarity_matrix[candidate_idx, selected_global_indices]
                    tfidf_penalty = float(np.max(similarities_to_selected))
                
                if semantic_similarity_matrix is not None:
                    semantic_sims = semantic_similarity_matrix[candidate_idx, selected_global_indices]
                    semantic_penalty = float(np.max(semantic_sims))
                    diversity_penalty = diversity_alpha * semantic_penalty + (1 - diversity_alpha) * tfidf_penalty
                else:
                    diversity_penalty = tfidf_penalty
            
            # MMR score
            mmr_score = lambda_param * relevance_score - (1 - lambda_param) * diversity_penalty
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = i
        
        # Select the best item
        selected_indices.append(best_idx)
        selected_scores.append(best_mmr_score)
        remaining.remove(best_idx)
    
    # Convert to global indices
    selected_global_indices = np.array([candidate_indices[i] for i in selected_indices])
    mmr_scores = np.array(selected_scores)
    
    return selected_global_indices, mmr_scores


def recommend_with_mmr(
    df: pd.DataFrame,
    sim_matrix: np.ndarray,
    title: str,
    k: int = 10,
    lambda_param: float = 0.5,
    initial_pool_size: int = 100
) -> List[Tuple[str, float]]:
    """
    Generate recommendations using MMR for diversity.
    
    Args:
        df: DataFrame with movie data
        sim_matrix: Similarity matrix (N x N)
        title: Query movie title
        k: Number of recommendations to return
        lambda_param: MMR diversity parameter (0-1)
        initial_pool_size: Size of initial candidate pool to consider
    
    Returns:
        List of (title_with_year, mmr_score) tuples
    """
    # Find query index
    title_to_idx = {t: i for i, t in enumerate(df["title"].tolist())}
    if title not in title_to_idx:
        raise ValueError(f"Title '{title}' not found.")
    
    query_idx = title_to_idx[title]
    
    # Get initial candidate pool (top-N by relevance)
    sims = sim_matrix[query_idx].copy()
    sims[query_idx] = -1.0  # Exclude query itself
    
    # Get top candidates
    pool_size = min(initial_pool_size, len(sims))
    candidate_indices = np.argsort(sims)[::-1][:pool_size]
    candidate_scores = sims[candidate_indices]
    
    # Apply MMR re-ranking
    selected_indices, mmr_scores = mmr_rerank(
        query_idx=query_idx,
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
        similarity_matrix=sim_matrix,
        k=k,
        lambda_param=lambda_param
    )
    
    # Convert to titles
    if "title_with_year" in df.columns:
        recommendations = [
            (df.iloc[idx]["title_with_year"], float(score))
            for idx, score in zip(selected_indices, mmr_scores)
        ]
    else:
        recommendations = [
            (df.iloc[idx]["title"], float(score))
            for idx, score in zip(selected_indices, mmr_scores)
        ]
    
    return recommendations


def genre_diversity_filter(
    recommendations: List[Tuple[str, float]],
    df: pd.DataFrame,
    max_per_genre: int = 3
) -> List[Tuple[str, float]]:
    """
    Filter recommendations to ensure genre diversity.
    
    Limits the number of movies from the same primary genre.
    
    Args:
        recommendations: List of (title, score) tuples
        df: DataFrame with movie data including genres
        max_per_genre: Maximum movies allowed per genre
    
    Returns:
        Filtered list of recommendations
    """
    if "genres" not in df.columns:
        return recommendations
    
    # Track genre counts
    genre_counts = {}
    filtered_recs = []
    
    # Create title lookup
    title_to_genres = {}
    for _, row in df.iterrows():
        title = row.get("title_with_year", row["title"])
        # Parse first genre
        genres_str = str(row.get("genres", ""))
        if "[" in genres_str:
            # Extract first genre from JSON-like string
            try:
                import ast
                genres_list = ast.literal_eval(genres_str)
                if genres_list and isinstance(genres_list, list):
                    primary_genre = genres_list[0].get("name", "Unknown")
                    title_to_genres[title] = primary_genre
            except:
                title_to_genres[title] = "Unknown"
        else:
            title_to_genres[title] = "Unknown"
    
    # Filter based on genre diversity
    for title, score in recommendations:
        genre = title_to_genres.get(title, "Unknown")
        
        if genre not in genre_counts:
            genre_counts[genre] = 0
        
        if genre_counts[genre] < max_per_genre:
            filtered_recs.append((title, score))
            genre_counts[genre] += 1
    
    return filtered_recs


def compute_diversity_metrics(
    recommendations: List[Tuple[str, float]],
    similarity_matrix: np.ndarray,
    df: pd.DataFrame
) -> dict:
    """
    Compute diversity metrics for a recommendation list.
    
    Args:
        recommendations: List of (title, score) tuples
        similarity_matrix: Full similarity matrix
        df: DataFrame with movie data
    
    Returns:
        Dictionary with diversity metrics
    """
    if len(recommendations) < 2:
        return {
            "avg_pairwise_distance": 0.0,
            "min_pairwise_distance": 0.0,
            "coverage": 0.0
        }
    
    # Get indices
    title_to_idx = {t: i for i, t in enumerate(df.get("title_with_year", df["title"]).tolist())}
    rec_titles = [title for title, _ in recommendations]
    rec_indices = [title_to_idx.get(title, -1) for title in rec_titles]
    rec_indices = [i for i in rec_indices if i >= 0]
    
    if len(rec_indices) < 2:
        return {
            "avg_pairwise_distance": 0.0,
            "min_pairwise_distance": 0.0,
            "coverage": 0.0
        }
    
    # Compute pairwise distances (1 - similarity)
    distances = []
    for i in range(len(rec_indices)):
        for j in range(i + 1, len(rec_indices)):
            idx_i, idx_j = rec_indices[i], rec_indices[j]
            similarity = similarity_matrix[idx_i, idx_j]
            distance = 1.0 - similarity
            distances.append(distance)
    
    return {
        "avg_pairwise_distance": float(np.mean(distances)),
        "min_pairwise_distance": float(np.min(distances)),
        "max_pairwise_distance": float(np.max(distances)),
        "num_items": len(rec_indices)
    }

