import ast
import re
from typing import List, Optional

import pandas as pd


def _safe_literal_eval(x) -> list:
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def parse_name_list(x, key: str = "name", top_n: Optional[int] = None) -> List[str]:
    arr = _safe_literal_eval(x)
    names = [d.get(key, "") for d in arr if isinstance(d, dict) and key in d]
    names = [n for n in names if n]
    if top_n is not None:
        names = names[:top_n]
    return names


def extract_directors(crew_json) -> List[str]:
    arr = _safe_literal_eval(crew_json)
    directors = []
    for d in arr:
        if isinstance(d, dict) and d.get("job") == "Director":
            name = d.get("name")
            if name and name not in directors:
                directors.append(name)
    return directors


def normalize_tokens_collapse_space(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        t = re.sub(r"\s+", " ", str(t)).strip().lower()
        t = t.replace(" ", "_")
        if t:
            out.append(t)
    return out


def _dedupe_preserve_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_tags_row(genres, keywords, cast, director) -> str:
    g = normalize_tokens_collapse_space(parse_name_list(genres))
    kw = normalize_tokens_collapse_space(parse_name_list(keywords))
    c = normalize_tokens_collapse_space(parse_name_list(cast, top_n=3))
    d = normalize_tokens_collapse_space(extract_directors(director))
    tokens = _dedupe_preserve_order(g + kw + c + d)
    return " ".join(tokens)


def build_enriched_text(title: str, overview: str, genres, keywords=None) -> str:
    parts: List[str] = []

    if title and str(title).strip():
        parts.append(str(title).strip())

    if overview and str(overview).strip():
        parts.append(str(overview).strip())

    genre_list = parse_name_list(genres)
    if genre_list:
        parts.append(f"Genres: {', '.join(genre_list)}")

    kw_list = parse_name_list(keywords) if keywords is not None else []
    if kw_list:
        parts.append(f"Keywords: {', '.join(kw_list)}")

    return ". ".join(parts)


def _require_columns(df: pd.DataFrame, cols: List[str], where: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        loc = f" in {where}" if where else ""
        raise ValueError(f"Missing required columns{loc}: {missing}")


def load_and_preprocess(movies_csv: str, credits_csv: str) -> pd.DataFrame:
    movies = pd.read_csv(movies_csv)
    credits = pd.read_csv(credits_csv)

    _require_columns(movies, ["id", "title", "overview", "release_date"], "movies_csv")
    _require_columns(credits, ["movie_id", "cast", "crew"], "credits_csv")

    df = movies.merge(credits, left_on="id", right_on="movie_id", suffixes=("_m", "_c"))
    df = df.drop(columns=["movie_id"])

    df["tags"] = [
        build_tags_row(genres, keywords, cast, crew)
        for genres, keywords, cast, crew in zip(
            df["genres"],
            df.get("keywords", pd.Series([[]] * len(df))),
            df["cast"],
            df["crew"],
        )
    ]

    df["overview_text"] = df["overview"].fillna("").astype(str)

    df["enriched_text"] = [
        build_enriched_text(title, overview, genres, keywords)
        for title, overview, genres, keywords in zip(
            df["title_m"],
            df["overview_text"],
            df["genres"],
            df.get("keywords", pd.Series([[]] * len(df))),
        )
    ]

    df["year"] = (
        pd.to_datetime(df["release_date"], errors="coerce").dt.year.fillna(0).astype(int)
    )

    df["title"] = df["title_m"].astype(str)
    df["title_with_year"] = df.apply(
        lambda row: f"{row['title']} ({int(row['year'])})" if row["year"] > 0 else row["title"],
        axis=1,
    )

    keep_cols = [
        "id",
        "title",
        "title_with_year",
        "year",
        "overview_text",
        "tags",
        "enriched_text",
    ]
    out = df[keep_cols].rename(columns={"id": "movie_id"})
    return out
