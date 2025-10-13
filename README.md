# 🎬 Movie Recommendation System

A production-ready content-based movie recommendation system with semantic understanding and fast similarity search.

## ✨ Features

- **Hybrid Recommendations** - Combines semantic (BERT) and metadata (TF-IDF) approaches
- **MMR Diversity** - Maximal Marginal Relevance for diverse recommendations
- **Configurable** - Easy parameter tuning via YAML config
- **Evaluation Metrics** - Objective quality measurement

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
python -m scripts.run_pipeline \
  --movies datasets/tmdb_5000_movies.csv \
  --credits datasets/tmdb_5000_credits.csv \
  --outdir outputs
```


## 📊 Performance

## 🔌 OpenSearch Setup

Set environment variables before running:

```bash
export OPENSEARCH_HOST="your-domain.us-east-1.es.amazonaws.com"
# Optional basic auth:
# export OPENSEARCH_BASIC_AUTH="username:password"
```

The pipeline will create the index and ingest embeddings automatically.

| Metric | Value |
|--------|-------|
| **Pipeline Execution** | ~10s (with cache) |
| **Recommendation Lookup** | < 1ms (cached) |
| **Memory Usage** | ~20 MB (cached) |
| **Scalability** | Millions of movies |

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
model:
  name: "all-MiniLM-L6-v2"  # Sentence transformer model
  batch_size: 32

similarity:
  alpha: 0.5  # BERT vs TF-IDF weight (0-1)

recommendations:
  default_k: 10
```

## 🎯 Command-Line Options

```bash
# Basic usage
python -m scripts.run_pipeline \
  --movies <path> \
  --credits <path> \
  --outdir <path>



# With evaluation
python -m scripts.run_pipeline \
  --movies <path> \
  --credits <path> \
  --outdir <path> \
  --evaluate

# Tune alpha parameter
python -m scripts.run_pipeline \
  --movies <path> \
  --credits <path> \
  --outdir <path> \
  --skip_embeddings \
  --alpha 0.7
```

## 📁 Project Structure

```
paper/
├── config.yaml                # Configuration file
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── datasets/                  # Movie data
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
├── scripts/                   # Source code
│   ├── run_pipeline.py       # Main pipeline
│   ├── embeddings.py         # BERT/sentence-transformers
│   ├── similarity.py         # Similarity computation
│   ├── evaluation.py         # Metrics
│   ├── recommend.py          # Recommendation logic
│   ├── utils.py              # Data preprocessing
│   ├── config.py             # Config loader
│   └── opensearch_store.py   # OpenSearch vector store
├── outputs/                   # Generated files
│   ├── preprocessed.pkl
│   ├── bert_embeddings.npy
│   └── recommendations_*.json
```

## 🔧 How It Works

### 1. Data Preprocessing
- Merges movie metadata and credits
- Extracts features: genres, keywords, cast, director
- Creates enriched text: `"Title. Overview. Genres: ... Keywords: ..."`

### 2. Feature Extraction
- **Semantic:** BERT embeddings from enriched text (384-dim)
- **Metadata:** TF-IDF vectors from tags (50K-dim)

### 3. Similarity Computation
- **BERT:** Cosine similarity on embeddings
- **TF-IDF:** Cosine similarity on sparse vectors
- **Hybrid:** Weighted combination (α × BERT + (1-α) × TF-IDF)

### 4. Recommendations
- Find top-K most similar movies

## 📈 Example Results

**Recommendations for "The Dark Knight":**
1. The Dark Knight Rises (2012) - 0.5909
2. Batman Begins (2005) - 0.4745
3. Batman (1989) - 0.4516
4. Batman v Superman: Dawn of Justice (2016) - 0.4275
5. Batman Returns (1992) - 0.4093

**Recommendations for "Inception":**
1. Subconscious (2015) - 0.2984
2. In Dreams (1999) - 0.2834
3. Unknown (2011) - 0.2808
4. Looper (2012) - 0.2739
5. Frailty (2001) - 0.2660

## 🧪 Evaluation Metrics

Run with `--evaluate` to get:
- **Diversity** - Variety of recommendations
- **Coverage** - % of catalog recommended
- **Intra-list Diversity** - Difference within recommendation lists
- **Novelty** - Frequency of recommending less popular items

## 🎓 Technical Details

### Models Used
- **Default:** `all-MiniLM-L6-v2` (22M params, 384-dim)
- **Alternative:** `all-mpnet-base-v2` (110M params, 768-dim, higher quality)

### Algorithms
- **Similarity:** Cosine similarity on L2-normalized vectors
- **Fusion:** Linear combination of semantic + metadata signals

### Scalability
- Designed to work with AWS OpenSearch for semantic kNN search
- Memory usage focused on TF-IDF and embeddings artifacts

## 🚦 Dependencies

Core:
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `torch`, `sentence-transformers`
- `joblib`, `pyyaml`

Optional:
- `kagglehub` - For dataset download

## 🤝 Contributing

This is a research/educational project. Feel free to:
- Experiment with different models
- Try different alpha values
- Add new evaluation metrics
- Implement Phase 3 features (MMR, API)

## 📝 License

This project uses the TMDB 5000 Movie Dataset for educational purposes.

## 🎯 Advanced Features (Phase 3)

### MMR Diversity
```bash
# Get diverse recommendations
python -m scripts.run_pipeline \
  --movies <path> \
  --credits <path> \
  --outdir <path> \
  --use_mmr \
  --mmr_lambda 0.5
```
---

**Built with:** Python, BERT, sentence-transformers, OpenSearch  
**Dataset:** TMDB 5000 Movie Dataset  
**Status:** Production Ready ✅

