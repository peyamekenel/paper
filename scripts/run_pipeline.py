"""
Movie Recommendation System - Unified Pipeline
Includes: Configuration, FAISS, Caching, and Evaluation
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
from scripts.similarity import cosine_sim_dense, cosine_sim_sparse
from scripts.recommend import recommend_top_k
from scripts.cache import RecommendationCache
from scripts.evaluation import evaluate_recommendations, print_evaluation_report
from scripts.diversity import recommend_with_mmr, compute_diversity_metrics
from scripts.learned_fusion import create_learned_similarity

# Try to import FAISS (optional)
try:
    from scripts.faiss_similarity import FAISSIndex, FAISS_AVAILABLE
except ImportError:
    FAISS_AVAILABLE = False


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
    parser.add_argument("--use_faiss", action="store_true", help="Enable FAISS")
    parser.add_argument("--use_cache", action="store_true", help="Enable caching")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument("--skip_embeddings", action="store_true", help="Use cached embeddings")
    
    # Phase 3 arguments
    parser.add_argument("--use_mmr", action="store_true", help="Enable MMR diversity")
    parser.add_argument("--mmr_lambda", type=float, default=None, help="MMR lambda (0-1)")
    parser.add_argument("--use_learned_fusion", action="store_true", help="Use learned fusion")
    parser.add_argument("--fusion_model", default=None, help="Fusion model type")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config.update_from_args(args)
    
    # Apply CLI overrides
    if args.use_faiss:
        config.set('similarity.use_faiss', True)
    if args.use_cache:
        config.set('performance.use_cache', True)
    if args.use_mmr:
        config.set('recommendations.use_mmr', True)
    if args.mmr_lambda is not None:
        config.set('recommendations.mmr_lambda', args.mmr_lambda)
    if args.use_learned_fusion:
        config.set('recommendations.use_learned_fusion', True)
    if args.fusion_model is not None:
        config.set('recommendations.fusion_model_type', args.fusion_model)
    
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
        vectorizer = TfidfVectorizer(
            max_features=config.get('tfidf.max_features', 50000),
            ngram_range=tuple(config.get('tfidf.ngram_range', [1, 2]))
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
    
    # ========== 4. SIMILARITY COMPUTATION ==========
    print(f"\n[4/6] Computing similarities...")
    
    use_faiss = config.get('similarity.use_faiss', False) and FAISS_AVAILABLE
    
    if use_faiss:
        print(f"  Using FAISS for fast search...")
        faiss_index = FAISSIndex(index_type=config.get('similarity.faiss_index_type', 'Flat'))
        faiss_index.build(bert_embs, verbose=False)
        
        # Compute top-K sparse matrix for compatibility
        from scripts.faiss_similarity import build_similarity_matrix_faiss
        sim_bert = build_similarity_matrix_faiss(bert_embs, k=100, verbose=False)
        faiss_index.save(os.path.join(args.outdir, "faiss_index.bin"))
    else:
        sim_bert = cosine_sim_dense(bert_embs)
        np.save(os.path.join(args.outdir, "sim_bert.npy"), sim_bert)
    
    sim_tfidf = cosine_sim_sparse(tfidf)
    np.save(os.path.join(args.outdir, "sim_tfidf.npy"), sim_tfidf)
    
    # Combine similarities
    use_learned_fusion = config.get('recommendations.use_learned_fusion', False)
    
    if use_learned_fusion:
        print(f"  Using learned fusion (ML-based combination)...")
        fusion_model_type = config.get('recommendations.fusion_model_type', 'gbm')
        fusion_samples = config.get('recommendations.fusion_train_samples', 10000)
        
        sim_combined, fusion_metrics = create_learned_similarity(
            sim_bert, sim_tfidf, df,
            model_type=fusion_model_type,
            sample_size=fusion_samples,
            save_path=os.path.join(args.outdir, "fusion_model.pkl"),
            verbose=True
        )
        
        # Save metrics
        with open(os.path.join(args.outdir, "fusion_metrics.json"), 'w') as f:
            json.dump(fusion_metrics, f, indent=2)
        
        print(f"  ✓ Learned fusion complete (Test R²={fusion_metrics['test_r2']:.4f})")
    else:
        alpha = config.get('similarity.alpha', 0.5)
        sim_combined = alpha * sim_bert + (1 - alpha) * sim_tfidf
        print(f"  ✓ Fixed fusion (α={alpha})")
    
    np.save(os.path.join(args.outdir, "sim_combined.npy"), sim_combined)
    
    # ========== 5. CACHING ==========
    cache = None
    if config.get('performance.use_cache', False):
        print(f"\n[5/6] Building recommendation cache...")
        cache = RecommendationCache(cache_dir=cache_dir)
        cache_top_k = config.get('recommendations.cache_top_k', 100)
        cache.build(df, sim_combined, top_k=cache_top_k, verbose=False)
        cache.save(prefix="recommendation_cache")
        
        stats = cache.get_statistics()
        print(f"  ✓ Cached {stats['num_movies']} movies with top-{stats['recommendations_per_movie']}")
        
        # Export sample
        cache.export_json(os.path.join(args.outdir, "cache_sample.json"), max_recommendations=10)
    else:
        print(f"\n[5/6] Caching disabled (use --use_cache to enable)")
    
    # ========== 6. EVALUATION ==========
    if args.evaluate:
        print(f"\n[6/6] Running evaluation...")
        test_titles = ["The Dark Knight", "Inception", "Avatar", "Toy Story", "The Matrix"]
        test_cases = [{'title': t} for t in test_titles if t in df['title'].values]
        
        metrics = evaluate_recommendations(test_cases, sim_combined, df, k=10)
        print_evaluation_report(metrics)
        
        with open(os.path.join(args.outdir, "evaluation_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
    else:
        print(f"\n[6/6] Evaluation disabled (use --evaluate to enable)")
    
    # ========== DEMO RECOMMENDATIONS ==========
    print(f"\n{'='*70}")
    print(f"📽️  Recommendations for: '{args.example_title}'")
    print(f"{'='*70}")
    
    k = config.get('recommendations.default_k', 10)
    use_mmr = config.get('recommendations.use_mmr', False)
    
    # Generate recommendations
    if use_mmr:
        # Use MMR for diversity
        mmr_lambda = config.get('recommendations.mmr_lambda', 0.5)
        mmr_pool = config.get('recommendations.mmr_pool_size', 100)
        
        print(f"🎯 Using MMR diversity (λ={mmr_lambda})")
        
        recs_mmr = recommend_with_mmr(
            df, sim_combined, args.example_title,
            k=k, lambda_param=mmr_lambda, initial_pool_size=mmr_pool
        )
        
        # Also get standard recommendations for comparison
        recs_standard = recommend_top_k(df, sim_combined, args.example_title, k=k)
        
        recs = {
            "mmr": recs_mmr,
            "standard": recs_standard
        }
        
        # Compute diversity metrics
        diversity_mmr = compute_diversity_metrics(recs_mmr, sim_combined, df)
        diversity_standard = compute_diversity_metrics(recs_standard, sim_combined, df)
        
        print(f"  MMR diversity: {diversity_mmr['avg_pairwise_distance']:.4f}")
        print(f"  Standard diversity: {diversity_standard['avg_pairwise_distance']:.4f}")
        
    elif cache:
        # Use cache if available
        try:
            recs_combined = cache.get_recommendations(args.example_title, k=k)
            recs = {"hybrid": recs_combined}
            print(f"⚡ Using cached recommendations")
        except ValueError:
            recs = {
                "semantic": recommend_top_k(df, sim_bert, args.example_title, k=k),
                "metadata": recommend_top_k(df, sim_tfidf, args.example_title, k=k),
                "hybrid": recommend_top_k(df, sim_combined, args.example_title, k=k),
            }
    else:
        # Standard recommendations
        recs = {
            "semantic": recommend_top_k(df, sim_bert, args.example_title, k=k),
            "metadata": recommend_top_k(df, sim_tfidf, args.example_title, k=k),
            "hybrid": recommend_top_k(df, sim_combined, args.example_title, k=k),
        }
    
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
    print(f"  FAISS enabled: {use_faiss}")
    print(f"  Cache enabled: {cache is not None}")
    print(f"  MMR diversity: {use_mmr}")
    print(f"  Learned fusion: {use_learned_fusion}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
