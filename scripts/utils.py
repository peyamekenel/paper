import ast
import json
import os
import re
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd


def _safe_literal_eval(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def parse_name_list(x, key="name", top_n=None):
    arr = _safe_literal_eval(x)
    names = [d.get(key, "") for d in arr if isinstance(d, dict) and key in d]
    names = [n for n in names if n]
    if top_n is not None:
        names = names[:top_n]
    return names


def extract_director(crew_json):
    arr = _safe_literal_eval(crew_json)
    for d in arr:
        if isinstance(d, dict) and d.get("job") == "Director":
            name = d.get("name")
            if name:
                return [name]
    return []


def normalize_tokens(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        t = re.sub(r"\s+", "", str(t)).lower()
        if t:
            out.append(t)
    return out


def build_tags_row(genres, keywords, cast, director):
    g = normalize_tokens(parse_name_list(genres))
    kw = normalize_tokens(parse_name_list(keywords))
    c = normalize_tokens(parse_name_list(cast, top_n=3))
    d = normalize_tokens(extract_director(director))
    return " ".join(g + kw + c + d)


def build_enriched_text(title: str, overview: str, genres, keywords=None) -> str:
    """
    Build enriched text for semantic embeddings by combining title, overview, and metadata.
    
    This creates a natural language representation that helps BERT/sentence-transformers
    better understand the movie's content and context.
    
    Args:
        title: Movie title
        overview: Movie plot overview/description
        genres: Genres (JSON string or list)
        keywords: Optional keywords (JSON string or list)
    
    Returns:
        Enriched text string like: "Title. Overview. Genres: action, adventure. Keywords: hero, save world"
    """
    parts = []
    
    # Add title
    if title and str(title).strip():
        parts.append(str(title).strip())
    
    # Add overview
    if overview and str(overview).strip():
        parts.append(str(overview).strip())
    
    # Add genres in natural language
    genre_list = parse_name_list(genres)
    if genre_list:
        # Keep original case for genres (looks more natural)
        parts.append(f"Genres: {', '.join(genre_list).lower()}")
    
    # Add keywords if provided
    if keywords:
        keyword_list = parse_name_list(keywords, top_n=10)
        if keyword_list:
            parts.append(f"Keywords: {', '.join(keyword_list).lower()}")
    
    return ". ".join(parts)


def load_and_preprocess(movies_csv: str, credits_csv: str) -> pd.DataFrame:
    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv)

    # Merge on movie ID to avoid duplicate titles causing duplicate rows
    df = movies.merge(credits, left_on="id", right_on="movie_id", suffixes=("_m", "_c"))
    df["tags"] = [
        build_tags_row(genres, keywords, cast, crew)
        for genres, keywords, cast, crew in zip(
            df["genres"], 
            df["keywords"],
            df["cast"],
            df["crew"],
        )
    ]

    df["overview_text"] = df["overview"].fillna("").astype(str)

    # Generate enriched text for semantic embeddings (title + overview + genres + keywords)
    # Use title_m (from movies dataset) as the canonical title
    df["enriched_text"] = [
        build_enriched_text(title, overview, genres, keywords)
        for title, overview, genres, keywords in zip(
            df["title_m"],
            df["overview_text"],
            df["genres"],
            df["keywords"]
        )
    ]

    # Extract year from release_date for display purposes
    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)

    # Create display title with year (e.g., "Batman (1989)")
    # Use title_m as the canonical title (from movies dataset)
    df["title"] = df["title_m"]
    df["title_with_year"] = df.apply(
        lambda row: f"{row['title']} ({int(row['year'])})" if row['year'] > 0 else row['title'],
        axis=1
    )

    keep_cols = ["id", "title", "title_with_year", "year", "overview_text", "tags", "enriched_text"]
    return df[keep_cols].rename(columns={"id": "movie_id"})
