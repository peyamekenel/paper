"""
Movie Recommendation System - Unified Pipeline
OpenSearch-only backend with legacy features removed.
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.config import load_config
from scripts.utils import load_and_preprocess
from scripts.embeddings import BertEmbedder
from scripts.similarity import minmax_normalize
from scripts.recommend import recommend_top_k
from scripts.evaluation import evaluate_recommendations, print_evaluation_report
from scripts.diversity import recommend_with_mmr, compute_diversity_metrics
from scripts.opensearch_store import OpenSearchVectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Movie Recommendation System - Production Pipeline"
    )
    parser.add_argument("--movies", required=True, help="Path to movies CSV file")
    parser.add_argument("--credits", required=True, help="Path to credits CSV file")
    parser.add_argument("--outdir", required=True, help="Output directory for results")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--model", default=None, help="Model name (overrides config)")
    parser.add_argument("--k", type=int, default=None, help="Number of recommendations")
    parser.add_argument("--alpha", type=float, default=None, help="BERT/TF-IDF weight")
    parser.add_argument("--example_title", default="The Dark Knight", help="Example movie")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--skip_embeddings", action="store_true", help="Use cached embeddings")
    
    # Diversity arguments
    parser.add_argument("--use_mmr", action="store_true", help="Enable MMR diversity")
    parser.add_argument("--mmr_lambda", type=float, default=None, help="MMR lambda (0-1)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config.update_from_args(args)
    
    # Apply CLI overrides
    if args.use_mmr:
        config.set('recommendations.use_mmr', True)
    if args.mmr_lambda is not None:
        config.set('recommendations.mmr_lambda', args.mmr_lambda)
    
    # Create directories
    os.makedirs(args.outdir, exist_ok=True)
    cache_dir = config.get('paths.cache_dir', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    start_time = time.time()
    
    print("="*70)
    print("🎬 MOVIE RECOMMENDATION SYSTEM")
    print("="*70)
    
    # ========== 1. DATA LOADING ==========
    print(f"\n[1/6] Loading and preprocessing data...")
    df = load_and_preprocess(args.movies, args.credits)
    df = df.dropna(subset=["enriched_text", "tags"])
    df = df.reset_index(drop=True)
    print(f"  ✓ Loaded {len(df)} movies")
    joblib.dump(df, os.path.join(args.outdir, "preprocessed.pkl"))
    
    # ========== 2. TF-IDF FEATURES ==========
    print(f"\n[2/6] Building TF-IDF features...")
    tfidf_path = os.path.join(args.outdir, "tfidf_matrix.npz")
    
    if os.path.exists(tfidf_path) and args.skip_embeddings:
        print(f"  Loading cached TF-IDF...")
        tfidf = sparse.load_npz(tfidf_path)
        vectorizer = joblib.load(os.path.join(args.outdir, "tfidf_vectorizer.joblib"))
    else:
        ngram_range_cfg = config.get('tfidf.ngram_range', [1, 2])
        ngram_range = (int(ngram_range_cfg[0]), int(ngram_range_cfg[1]))
        vectorizer = TfidfVectorizer(
            max_features=config.get('tfidf.max_features', 50000),
            ngram_range=ngram_range
        )
        tfidf = vectorizer.fit_transform(df["tags"].tolist())
        sparse.save_npz(tfidf_path, tfidf)
        joblib.dump(vectorizer, os.path.join(args.outdir, "tfidf_vectorizer.joblib"))
    
    print(f"  ✓ TF-IDF shape: {tfidf.shape}")
    
    # ========== 3. SEMANTIC EMBEDDINGS ==========
    print(f"\n[3/6] Encoding semantic embeddings...")
    embeddings_path = os.path.join(args.outdir, "bert_embeddings.npy")
    
    if os.path.exists(embeddings_path) and args.skip_embeddings:
        print(f"  Loading cached embeddings...")
        bert_embs = np.load(embeddings_path)
    else:
        model_name = config.get('model.name', 'all-MiniLM-L6-v2')
        batch_size = config.get('model.batch_size', 32)
        max_length = config.get('model.max_length', 256)
        
        print(f"  Model: {model_name}")
        be = BertEmbedder(model_name)
        bert_embs = be.encode_cls(
            df["enriched_text"].tolist(), 
            batch_size=batch_size, 
            max_length=max_length
        )
        np.save(embeddings_path, bert_embs)
    
    print(f"  ✓ Embeddings shape: {bert_embs.shape}")
    
    # ========== 4. SKIP NxN SIMILARITY COMPUTATION ==========
    print(f"\n[4/6] Skipping full NxN similarity matrix (computed on-demand per query)")
    
    # ========== 5. EVALUATION ==========
    if args.evaluate:
        print(f"\n[5/6] Evaluation skipped (requires full similarity matrix)")
        print(f"  Note: Evaluation functionality removed for scalability")
    else:
        print(f"\n[5/6] Evaluation disabled (use --evaluate to enable)")
    
    # ========== DEMO RECOMMENDATIONS ==========
    print(f"\n{'='*70}")
    print(f"📽️  Recommendations for: '{args.example_title}'")
    print(f"{'='*70}")
    
    k = config.get('recommendations.default_k', 10)
    use_mmr = config.get('recommendations.use_mmr', False)

    host = os.getenv("OPENSEARCH_HOST", config.get("opensearch.host", ""))
    if not host:
        raise RuntimeError("OpenSearch host not configured. Set OPENSEARCH_HOST in .env file or opensearch.host in config.")
    port = int(os.getenv("OPENSEARCH_PORT", config.get("opensearch.port", 443)))
    index_name = os.getenv("OPENSEARCH_INDEX_NAME", config.get("opensearch.index_name", "movies"))
    auth_env = os.getenv("OPENSEARCH_BASIC_AUTH", "")
    http_auth = tuple(auth_env.split(":", 1)) if auth_env else None

    os_store = OpenSearchVectorStore(
        host=host,
        port=port,
        index_name=index_name,
        embedding_dim=bert_embs.shape[1],
        http_auth=http_auth,
    )
    os_store.ensure_index()
    os_store.index_documents(df, bert_embs)

    # Generate recommendations via OpenSearch + TF-IDF hybrid
    matches = df.index[df['title'] == args.example_title].tolist()
    if not matches:
        raise ValueError(f"Title '{args.example_title}' not found.")
    q = matches[0]

    mmr_pool_size = config.get('recommendations.mmr_pool_size', 100)
    knn = os_store.knn_query_by_id(q, k=max(k, mmr_pool_size))
    cand_indices = np.array([idx for idx, _ in knn if idx != q])
    sem_sims = np.array([score for idx, score in knn if idx != q])[: len(cand_indices)]

    q_vec = tfidf[q]
    tfidf_cand = tfidf[cand_indices]
    tfidf_scores = (tfidf_cand @ q_vec.T).toarray().ravel()

    sem_sims_norm = minmax_normalize(sem_sims)
    tfidf_norm = minmax_normalize(tfidf_scores)

    alpha = config.get("similarity.alpha", 0.5)
    hybrid_scores = alpha * sem_sims_norm + (1 - alpha) * tfidf_norm
    
    if use_mmr:
        from scripts.diversity import mmr_rerank
        
        cand_tfidf_sim = (tfidf_cand @ tfidf_cand.T).toarray()
        
        mmr_lambda = config.get('recommendations.mmr_lambda', 0.5)
        selected_local_indices, mmr_scores = mmr_rerank(
            query_idx=0,
            candidate_indices=np.arange(len(cand_indices)),
            candidate_scores=hybrid_scores,
            similarity_matrix=cand_tfidf_sim,
            k=k,
            lambda_param=mmr_lambda
        )
        
        mmr_global_indices = cand_indices[selected_local_indices]
        recs_mmr = [(df["title"].iloc[idx], float(score)) for idx, score in zip(mmr_global_indices, mmr_scores)]
        
        top_order = np.argsort(hybrid_scores)[::-1][:k]
        recs_hybrid = [(df["title"].iloc[cand_indices[i]], float(hybrid_scores[i])) for i in top_order]
        
        recs = {"hybrid": recs_hybrid, "mmr": recs_mmr}
    else:
        top_order = np.argsort(hybrid_scores)[::-1][:k]
        recs_hybrid = [(df["title"].iloc[cand_indices[i]], float(hybrid_scores[i])) for i in top_order]
        recs = {"hybrid": recs_hybrid}
    
    # Save recommendations
    output_file = os.path.join(args.outdir, f"recommendations_{args.example_title.replace(' ', '_')}.json")
    with open(output_file, "w") as f:
        json.dump(recs, f, indent=2)
    
    # Display results
    print(f"\nTop {min(5, k)} recommendations:")
    rec_list = recs.get("mmr", recs.get("hybrid", []))
    for i, (title, score) in enumerate(rec_list[:5], 1):
        print(f"  {i}. {title} (score: {score:.4f})")
    
    # ========== SUMMARY ==========
    elapsed_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ Pipeline Complete!")
    print(f"{'='*70}")
    print(f"  Movies processed: {len(df)}")
    print(f"  Output directory: {args.outdir}")
    print(f"  Recommendations: {output_file}")
    print(f"  Execution time: {elapsed_time:.1f}s")
    print(f"  MMR diversity: {use_mmr}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
