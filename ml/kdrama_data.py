"""K-Drama AI - data loading and feature engineering.

This single file builds BOTH tables used in the project:

* the DRAMA table   (one row = one K-Drama)            -> Project 2, website catalogue
* the REVIEW table  (one row = one viewer x one drama) -> Project 1

Everything comes from two public Kaggle datasets that were scraped from
MyDramaList (MDL). Nothing in these tables is simulated.

Source A  Chanon Charuchinda (2023), "Korean Drama from 2015-2023 with Actors & Reviews",
          Kaggle, licence CC0: Public Domain.
          https://www.kaggle.com/datasets/chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist
          files used: korean_drama.csv (1,752 dramas), reviews.csv (10,625 viewer reviews)

Source B  Noor Rizki (2023), "Top Korean Drama List (~1500)", Kaggle, licence not stated ("Unknown").
          https://www.kaggle.com/datasets/noorrizki/top-korean-drama-list-1500
          file used: kdrama_list.csv (1,647 dramas with genres and the public MDL score)
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/"
SOURCE_A = "chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist"
SOURCE_B = "noorrizki/top-korean-drama-list-1500"
RAW_FILES = {
    "korean_drama.csv": SOURCE_A,
    "reviews.csv": SOURCE_A,
    "kdrama_list.csv": SOURCE_B,
}

LIKE_THRESHOLD = 8.0          # a viewer "likes" a drama when their own overall score is >= 8 / 10
MISSING_RANK_CODE = 99999     # Source A writes 99999 when MDL shows no rank; we treat it as missing


def download_raw(raw_dir="data/raw") -> Path:
    """Download the three raw CSV files (only the ones that are not on disk yet)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for slug in sorted({s for f, s in RAW_FILES.items() if not (raw_dir / f).exists()}):
        request = urllib.request.Request(KAGGLE_DOWNLOAD + slug, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(request, timeout=120) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for name in archive.namelist():
            if name in RAW_FILES:
                (raw_dir / name).write_bytes(archive.read(name))
    return raw_dir


def _key(text) -> str:
    """Normalise a title so that the two sources can be joined ('Move to Heaven' -> 'movetoheaven')."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def build_drama_table(raw_dir="data/raw") -> pd.DataFrame:
    """One row per drama, with real MDL metadata and a drama-level genre score."""
    raw_dir = Path(raw_dir)
    meta = pd.read_csv(raw_dir / "korean_drama.csv")
    genre_list = pd.read_csv(raw_dir / "kdrama_list.csv")

    meta["join_key"] = meta["drama_name"].map(_key) + meta["year"].astype(str)
    genre_list["join_key"] = genre_list["Name"].map(_key) + genre_list["Year"].astype(str)
    df = meta.merge(genre_list[["join_key", "Genre", "Score"]], on="join_key", how="inner")

    dramas = pd.DataFrame({
        "drama_id": df["kdrama_id"],
        "title": df["drama_name"].str.strip(),
        "year": df["year"].astype(int),
        "genres": df["Genre"].map(lambda g: "|".join(x.strip() for x in str(g).split(",") if x.strip())),
        "runtime_min": (df["duration"] / 60).round(1),               # seconds -> minutes per episode
        "episodes": df["tot_eps"].astype(float),
        "popularity_rank": df["pop"].where(df["pop"] != MISSING_RANK_CODE).astype(float),
        "public_rating": df["Score"].astype(float),                   # MDL community average, 0-10
    })

    # Drama-level genre score: how well-rated the drama's genres are on MDL in general.
    # Step 1: average public rating of every genre. Step 2: average those values over the drama's genres.
    exploded = dramas.assign(genre=dramas["genres"].str.split("|")).explode("genre")
    genre_mean = exploded.groupby("genre")["public_rating"].mean()
    dramas["genre_score"] = (
        exploded["genre"].map(genre_mean).groupby(level=0).mean().round(3)
    )
    return dramas.sort_values("popularity_rank", na_position="last").reset_index(drop=True)


def build_review_table(drama_table: pd.DataFrame, raw_dir="data/raw") -> pd.DataFrame:
    """One row per real viewer review, with the binary target and a viewer-level genre score."""
    reviews = pd.read_csv(Path(raw_dir) / "reviews.csv",
                          usecols=["user_id", "title", "overall_score"])
    dramas = drama_table.assign(title_key=drama_table["title"].map(_key))
    dramas = dramas.drop_duplicates("title_key", keep=False)          # drop ambiguous (repeated) titles
    reviews["title_key"] = reviews["title"].map(_key)
    df = reviews.merge(dramas.drop(columns=["title", "genre_score"]), on="title_key", how="inner")
    df = df.drop_duplicates(["user_id", "drama_id"]).reset_index(drop=True)

    # Viewer-level genre score = the mean score the SAME viewer gave to OTHER dramas that share
    # at least one genre with this drama. The review being predicted is never included, so the
    # feature does not contain the target. No other matching review -> missing value (NaN).
    genre_sets = df["genres"].str.split("|").map(set)
    scores = df["overall_score"].to_numpy()
    genre_score = np.full(len(df), np.nan)
    for _, idx in df.groupby("user_id").indices.items():
        if len(idx) < 2:
            continue
        for i in idx:
            others = [scores[j] for j in idx if j != i and genre_sets[i] & genre_sets[j]]
            if others:
                genre_score[i] = np.mean(others)

    out = pd.DataFrame({
        "viewer_id": df["user_id"],
        "drama_id": df["drama_id"],
        "title": df["title"],
        "genres": df["genres"],
        "genre_score": np.round(genre_score, 3),
        "runtime_min": df["runtime_min"],
        "popularity_rank": df["popularity_rank"],
        "public_rating": df["public_rating"],
        "episodes": df["episodes"],
        "year": df["year"],
        "viewer_score": df["overall_score"],
        "liked": (df["overall_score"] >= LIKE_THRESHOLD).astype(int),
    })
    return out


def load_tables(raw_dir="data/raw"):
    """Download (if needed) and build both tables. Returns (drama_table, review_table)."""
    download_raw(raw_dir)
    dramas = build_drama_table(raw_dir)
    return dramas, build_review_table(dramas, raw_dir)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    drama_table, review_table = load_tables(root / "data" / "raw")
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    drama_table.to_csv(processed / "dramas.csv", index=False)
    review_table.to_csv(processed / "viewer_reviews.csv", index=False)
    print("dramas:", drama_table.shape, "| viewer reviews:", review_table.shape)
    print(drama_table.isna().sum().to_string())
    print(review_table.isna().sum().to_string())
    print("like rate:", round(review_table["liked"].mean(), 3))
