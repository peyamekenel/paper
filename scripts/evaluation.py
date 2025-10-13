"""Evaluation metrics for recommendation quality."""
from typing import List, Tuple, Set, Dict
import numpy as np
import pandas as pd
from collections import defaultdict


def calculate_diversity(recommendations: List[List[Tuple[str, float]]], 
                         df: pd.DataFrame) -> float:
    """
    Calculate diversity of recommendations using genre distribution.
    
    Higher diversity = recommendations cover more different genres.
    
    Args:
        recommendations: List of recommendation lists (each is list of (title, score))
        df: DataFrame with movie data including genres
    
    Returns:
        Diversity score (0-1, higher is better)
    """
    all_recommended_titles = set()
    for recs in recommendations:
        all_recommended_titles.update([title for title, _ in recs])
    
    if not all_recommended_titles:
        return 0.0
    
    # Count unique items recommended
    unique_items = len(all_recommended_titles)
    total_items = len(df)
    
    # Diversity is the ratio of unique recommended items to total items
    diversity = unique_items / min(total_items, len(recommendations) * 10)
    
    return diversity


def calculate_coverage(recommendations: List[List[Tuple[str, float]]], 
                        df: pd.DataFrame) -> float:
    """
    Calculate catalog coverage.
    
    What percentage of the catalog appears in at least one recommendation?
    
    Args:
        recommendations: List of recommendation lists
        df: DataFrame with all movies
    
    Returns:
        Coverage score (0-1, higher is better)
    """
    all_recommended_titles = set()
    for recs in recommendations:
        all_recommended_titles.update([title for title, _ in recs])
    
    all_titles = set(df['title_with_year'].tolist())
    
    coverage = len(all_recommended_titles) / len(all_titles)
    return coverage


def calculate_novelty(recommendations: List[List[Tuple[str, float]]], 
                      df: pd.DataFrame, 
                      popularity_col: str = 'vote_count') -> float:
    """
    Calculate novelty score.
    
    Novelty measures how often the system recommends less popular items.
    Higher novelty = recommending more obscure/niche items.
    
    Args:
        recommendations: List of recommendation lists
        df: DataFrame with movie data
        popularity_col: Column name for popularity metric
    
    Returns:
        Novelty score (higher means recommending less popular items)
    """
    if popularity_col not in df.columns:
        return 0.0
    
    # Get popularity scores
    title_to_popularity = dict(zip(df['title_with_year'], df[popularity_col]))
    
    # Calculate average popularity of recommended items
    recommended_popularities = []
    for recs in recommendations:
        for title, _ in recs:
            if title in title_to_popularity:
                recommended_popularities.append(title_to_popularity[title])
    
    if not recommended_popularities:
        return 0.0
    
    # Novelty is inverse of popularity (normalized)
    avg_popularity = np.mean(recommended_popularities)
    max_popularity = df[popularity_col].max()
    
    # Lower popularity = higher novelty
    novelty = 1.0 - (avg_popularity / max_popularity)
    return novelty


def calculate_intra_list_diversity(recommendations: List[Tuple[str, float]], 
                                     similarity_matrix: np.ndarray,
                                     df: pd.DataFrame) -> float:
    """
    Calculate diversity within a single recommendation list.
    
    Measures how different the recommended items are from each other.
    
    Args:
        recommendations: Single list of (title, score) recommendations
        similarity_matrix: Similarity matrix for all items
        df: DataFrame with movie data
    
    Returns:
        Diversity score (0-1, higher is better)
    """
    if len(recommendations) < 2:
        return 1.0
    
    # Get indices of recommended items
    title_to_idx = {t: i for i, t in enumerate(df['title_with_year'].tolist())}
    rec_indices = []
    
    for title, _ in recommendations:
        if title in title_to_idx:
            rec_indices.append(title_to_idx[title])
    
    if len(rec_indices) < 2:
        return 1.0
    
    # Calculate average pairwise dissimilarity
    dissimilarities = []
    for i in range(len(rec_indices)):
        for j in range(i + 1, len(rec_indices)):
            idx_i, idx_j = rec_indices[i], rec_indices[j]
            similarity = similarity_matrix[idx_i, idx_j]
            dissimilarity = 1.0 - similarity
            dissimilarities.append(dissimilarity)
    
    avg_dissimilarity = np.mean(dissimilarities)
    return avg_dissimilarity


def evaluate_recommendations(test_cases: List[Dict[str, any]], 
                              similarity_matrix: np.ndarray,
                              df: pd.DataFrame,
                              k: int = 10) -> Dict[str, float]:
    """
    Comprehensive evaluation of recommendation system.
    
    Args:
        test_cases: List of dicts with 'title' and optionally 'expected_recs'
        similarity_matrix: Similarity matrix for all movies
        df: DataFrame with movie data
        k: Number of recommendations to generate
    
    Returns:
        Dictionary of evaluation metrics
    """
    from scripts.recommend import recommend_top_k
    
    all_recommendations = []
    intra_diversities = []
    
    for test_case in test_cases:
        title = test_case['title']
        
        try:
            # Generate recommendations
            recs = recommend_top_k(df, similarity_matrix, title, k=k)
            all_recommendations.append(recs)
            
            # Calculate intra-list diversity
            diversity = calculate_intra_list_diversity(recs, similarity_matrix, df)
            intra_diversities.append(diversity)
            
        except ValueError:
            continue
    
    if not all_recommendations:
        return {}
    
    # Calculate metrics
    metrics = {
        'diversity': calculate_diversity(all_recommendations, df),
        'coverage': calculate_coverage(all_recommendations, df),
        'intra_list_diversity': np.mean(intra_diversities),
        'num_test_cases': len(all_recommendations),
    }
    
    # Add novelty if vote_count is available
    if 'vote_count' in df.columns:
        metrics['novelty'] = calculate_novelty(all_recommendations, df, 'vote_count')
    
    return metrics


def print_evaluation_report(metrics: Dict[str, float]) -> None:
    """Print formatted evaluation report."""
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name:30s}: {value:.4f}")
        else:
            print(f"{metric_name:30s}: {value}")
    
    print("="*60 + "\n")

