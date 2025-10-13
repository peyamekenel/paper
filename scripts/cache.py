"""Caching system for pre-computed recommendations."""
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


class RecommendationCache:
    """
    Cache for pre-computed movie recommendations.
    
    Stores top-K similar movies for each movie to enable instant lookups.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        self.similarity_cache: Dict[str, List[Tuple[int, float]]] = {}
        self.title_to_idx: Dict[str, int] = {}
        self.idx_to_title: Dict[int, str] = {}
        self.titles_with_year: List[str] = []
    
    def build(self, df: pd.DataFrame, similarity_matrix: np.ndarray, 
              top_k: int = 100, verbose: bool = True) -> None:
        """
        Build cache from similarity matrix.
        
        Args:
            df: DataFrame with movie data (must have 'title' and 'title_with_year')
            similarity_matrix: Similarity matrix (N x N), may be sparse
            top_k: Number of top similar movies to cache per movie
            verbose: Print progress
        """
        if verbose:
            print(f"Building recommendation cache (top-{top_k} per movie)...")
        
        n_movies = len(df)
        
        # Build title mappings using DataFrame index
        titles_list = df['title'].tolist()
        self.title_to_idx = {title: i for i, title in enumerate(titles_list)}
        self.titles_with_year = df['title_with_year'].tolist()
        
        # Pre-compute top-k for each movie
        for idx in range(n_movies):
            # Handle both dense and sparse matrices
            if hasattr(similarity_matrix, 'toarray'):
                # Sparse matrix
                sims = similarity_matrix[idx].toarray().flatten()
            else:
                # Dense matrix
                sims = similarity_matrix[idx].copy()
            
            # Exclude self
            sims[idx] = -1.0
            
            # Get top-k indices (only consider non-zero similarities)
            valid_indices = np.where(sims > 0)[0]
            if len(valid_indices) == 0:
                # Fallback: use all indices if no valid similarities
                valid_indices = np.arange(len(sims))
            
            valid_sims = sims[valid_indices]
            sorted_idx = np.argsort(valid_sims)[::-1][:top_k]
            top_indices = valid_indices[sorted_idx]
            top_sims = valid_sims[sorted_idx]
            
            # Store as list of (index, similarity) tuples using title from list
            title = titles_list[idx]
            self.similarity_cache[title] = [(int(i), float(s)) for i, s in zip(top_indices, top_sims)]
        
        if verbose:
            print(f"  Cached {len(self.similarity_cache)} movies with up to {top_k} recommendations each")
    
    def get_recommendations(self, title: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get recommendations for a movie from cache.
        
        Args:
            title: Movie title (without year)
            k: Number of recommendations to return
        
        Returns:
            List of (title_with_year, similarity) tuples
        """
        if title not in self.similarity_cache:
            raise ValueError(f"Title '{title}' not found in cache")
        
        cached_results = self.similarity_cache[title][:k]
        
        # Convert indices to titles with years
        recommendations = [
            (self.titles_with_year[idx], sim)
            for idx, sim in cached_results
        ]
        
        return recommendations
    
    def save(self, prefix: str = "recommendation_cache") -> None:
        """
        Save cache to disk.
        
        Args:
            prefix: Filename prefix for cache files
        """
        cache_file = os.path.join(self.cache_dir, f"{prefix}.pkl")
        
        cache_data = {
            'similarity_cache': self.similarity_cache,
            'title_to_idx': self.title_to_idx,
            'idx_to_title': self.idx_to_title,
            'titles_with_year': self.titles_with_year,
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Cache saved to {cache_file}")
    
    def load(self, prefix: str = "recommendation_cache") -> bool:
        """
        Load cache from disk.
        
        Args:
            prefix: Filename prefix for cache files
        
        Returns:
            True if cache loaded successfully, False otherwise
        """
        cache_file = os.path.join(self.cache_dir, f"{prefix}.pkl")
        
        if not os.path.exists(cache_file):
            return False
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.similarity_cache = cache_data['similarity_cache']
        self.title_to_idx = cache_data['title_to_idx']
        self.idx_to_title = cache_data['idx_to_title']
        self.titles_with_year = cache_data['titles_with_year']
        
        print(f"Cache loaded from {cache_file} ({len(self.similarity_cache)} movies)")
        return True
    
    def export_json(self, output_path: str, max_recommendations: int = 20) -> None:
        """
        Export cache to JSON format for easy inspection.
        
        Args:
            output_path: Path to output JSON file
            max_recommendations: Maximum recommendations to include per movie
        """
        export_data = {}
        
        for title, cached_recs in self.similarity_cache.items():
            title_with_year = self.titles_with_year[self.title_to_idx[title]]
            
            recommendations = [
                {
                    'title': self.titles_with_year[idx],
                    'similarity': float(sim)
                }
                for idx, sim in cached_recs[:max_recommendations]
            ]
            
            export_data[title_with_year] = recommendations
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Cache exported to {output_path}")
    
    def get_statistics(self) -> Dict[str, float]:
        """Get cache statistics."""
        if not self.similarity_cache:
            return {}
        
        all_similarities = []
        for cached_recs in self.similarity_cache.values():
            all_similarities.extend([sim for _, sim in cached_recs])
        
        return {
            'num_movies': len(self.similarity_cache),
            'recommendations_per_movie': len(next(iter(self.similarity_cache.values()))),
            'mean_similarity': float(np.mean(all_similarities)),
            'median_similarity': float(np.median(all_similarities)),
            'min_similarity': float(np.min(all_similarities)),
            'max_similarity': float(np.max(all_similarities)),
        }

