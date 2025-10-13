from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class BertEmbedder:
    """
    Embedder using sentence-transformers for efficient semantic encoding.
    Default model: all-MiniLM-L6-v2 (5x faster than bert-base, better quality)
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        self.model_name = model_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode_cls(self, texts: List[str], batch_size: int = 32, max_length: int = 256, show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into normalized embeddings.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding (increased from 16 to 32 due to smaller model)
            max_length: Maximum sequence length (tokens beyond this are truncated)
            show_progress: Whether to show progress bar
        
        Returns:
            numpy array of shape (len(texts), embedding_dim) with L2-normalized embeddings
        """
        # sentence-transformers handles batching, normalization, and progress internally
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
            convert_to_numpy=True
        )
        return embeddings
