"""FAISS-based similarity search for scalable recommendations."""
from typing import List, Tuple, Optional
import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FAISSIndex:
    """
    FAISS-based similarity index for fast nearest neighbor search.
    
    This replaces the full N×N similarity matrix computation with an efficient
    index that can scale to millions of movies.
    """
    
    def __init__(self, index_type: str = "Flat", use_gpu: bool = False):
        """
        Initialize FAISS index.
        
        Args:
            index_type: Type of FAISS index
                - "Flat": Exact search (IndexFlatIP)
                - "IVF": Approximate search (IndexIVFFlat) - faster for large datasets
            use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "FAISS is not installed. Install with: pip install faiss-cpu"
            )
        
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.index = None
        self.embeddings = None
        self.dimension = None
    
    def build(self, embeddings: np.ndarray, verbose: bool = True) -> None:
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: numpy array of shape (n_samples, embedding_dim)
                       Should be L2-normalized for cosine similarity
            verbose: Whether to print progress
        """
        n_samples, self.dimension = embeddings.shape
        self.embeddings = embeddings.astype(np.float32)
        
        if verbose:
            print(f"Building FAISS index ({self.index_type}) for {n_samples} samples...")
        
        if self.index_type == "Flat":
            # Exact search using inner product (cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(self.embeddings)
        
        elif self.index_type == "IVF":
            # Approximate search - faster for large datasets
            nlist = min(int(np.sqrt(n_samples)), 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train the index
            if verbose:
                print(f"  Training IVF index with {nlist} clusters...")
            self.index.train(self.embeddings)
            self.index.add(self.embeddings)
            
            # Set search parameters for recall/speed tradeoff
            self.index.nprobe = min(10, nlist)  # Number of clusters to search
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            if verbose:
                print("  Moving index to GPU...")
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        
        if verbose:
            print(f"  Index built successfully! Total vectors: {self.index.ntotal}")
    
    def search(self, query_idx: int, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k most similar items to the query.
        
        Args:
            query_idx: Index of the query item
            k: Number of neighbors to return
        
        Returns:
            Tuple of (similarities, indices)
            - similarities: Cosine similarity scores (for normalized vectors)
            - indices: Indices of k nearest neighbors
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        query_vector = self.embeddings[query_idx:query_idx+1]
        
        # Search for k+1 since query itself will be in results
        similarities, indices = self.index.search(query_vector, k + 1)
        
        # Remove the query itself from results
        mask = indices[0] != query_idx
        similarities = similarities[0][mask][:k]
        indices = indices[0][mask][:k]
        
        return similarities, indices
    
    def search_batch(self, query_indices: List[int], k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries.
        
        Args:
            query_indices: List of query indices
            k: Number of neighbors to return per query
        
        Returns:
            Tuple of (similarities, indices) arrays of shape (len(query_indices), k)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        query_vectors = self.embeddings[query_indices]
        
        # Search for k+1 since query itself will be in results
        similarities, indices = self.index.search(query_vectors, k + 1)
        
        # Remove query from its own results for each query
        result_sims = []
        result_indices = []
        
        for i, query_idx in enumerate(query_indices):
            mask = indices[i] != query_idx
            result_sims.append(similarities[i][mask][:k])
            result_indices.append(indices[i][mask][:k])
        
        return np.array(result_sims), np.array(result_indices)
    
    def save(self, path: str) -> None:
        """Save FAISS index to disk."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        # Save index (move from GPU to CPU first if needed)
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_cpu, path)
        else:
            faiss.write_index(self.index, path)
    
    def load(self, path: str, embeddings: np.ndarray) -> None:
        """
        Load FAISS index from disk.
        
        Args:
            path: Path to saved index
            embeddings: Original embeddings (needed for search)
        """
        self.index = faiss.read_index(path)
        self.embeddings = embeddings.astype(np.float32)
        self.dimension = embeddings.shape[1]
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)


def build_similarity_matrix_faiss(embeddings: np.ndarray, k: Optional[int] = None, 
                                   verbose: bool = True) -> np.ndarray:
    """
    Build similarity matrix using FAISS for efficiency.
    
    If k is provided, returns sparse top-k similarities.
    Otherwise, returns full dense similarity matrix.
    
    Args:
        embeddings: L2-normalized embeddings
        k: If provided, only compute top-k similarities per item
        verbose: Print progress
    
    Returns:
        Similarity matrix (dense or sparse top-k)
    """
    faiss_index = FAISSIndex(index_type="Flat")
    faiss_index.build(embeddings, verbose=verbose)
    
    n_samples = len(embeddings)
    
    if k is None:
        # Return full similarity matrix
        similarities, _ = faiss_index.index.search(embeddings.astype(np.float32), n_samples)
        return similarities
    else:
        # Return top-k similarities
        if verbose:
            print(f"Computing top-{k} similarities for each of {n_samples} items...")
        
        similarities = np.zeros((n_samples, n_samples), dtype=np.float32)
        
        # Batch search for efficiency
        batch_size = 1000
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_indices = list(range(i, end_idx))
            
            batch_sims, batch_nn_indices = faiss_index.search_batch(batch_indices, k=k)
            
            for j, query_idx in enumerate(batch_indices):
                similarities[query_idx, batch_nn_indices[j]] = batch_sims[j]
        
        return similarities

