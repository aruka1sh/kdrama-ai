"""K-Drama AI - FastAPI backend.

Serves the REAL trained model and the REAL analysis results created by:
    python ml/train_supervised.py        -> artifacts/preference_model.joblib, supervised_results.json
    python ml/analyze_unsupervised.py    -> artifacts/unsupervised_results.json

Start:  uvicorn app.main:app --reload --port 8000      (from the backend/ folder)

If an artifact is missing, the endpoints answer with HTTP 503 and a clear message.
Nothing is simulated or hard-coded here.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app import catalog

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).resolve().parents[1] / "data" / "ml"      # copy made by scripts/bundle_artifacts.py


def _locate(env_name: str, development_path: Path, bundled_path: Path) -> Path:
    """Local development uses the fresh files written by ml/*.py; a deployment (where only the
    backend folder is shipped) uses the copies bundled inside backend/data/ml."""
    if os.environ.get(env_name):
        return Path(os.environ[env_name])
    return development_path if development_path.exists() else bundled_path


ARTIFACTS = _locate("KDRAMA_ARTIFACTS", ROOT / "artifacts", BUNDLE)
DRAMAS_CSV = _locate("KDRAMA_DRAMAS_CSV", ROOT / "data" / "processed" / "dramas.csv", BUNDLE / "dramas.csv")

DISCLOSURE = (
    "Classroom demonstration. The model was trained on real MyDramaList reviews written by "
    "self-selected reviewers. It estimates how similar reviewers scored similar dramas; it is not "
    "a validated personal prediction for you."
)

FEATURE_INFO = {
    "genre_score": {"label": "Your genre preference score", "unit": "1-10",
                    "help": "The average score you usually give to dramas of this genre."},
    "runtime_min": {"label": "Episode runtime", "unit": "minutes",
                    "help": "Length of one episode."},
    "popularity_rank": {"label": "Popularity rank", "unit": "1 = most popular",
                        "help": "MyDramaList popularity rank of the drama."},
    "public_rating": {"label": "Public rating", "unit": "0-10",
                      "help": "Average MyDramaList rating of the drama."},
    "episodes": {"label": "Episodes", "unit": "count", "help": "Number of episodes."},
    "year": {"label": "Release year", "unit": "year", "help": "Year the drama was released."},
}

app = FastAPI(title="K-Drama AI API", version="1.0.0",
              description="Real model predictions and analysis results for the K-Drama AI final project.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(catalog.router)   # real drama metadata (TMDB with a key, TVmaze without)


# --------------------------------------------------------------------------- artifact loading
def _missing(name: str) -> HTTPException:
    return HTTPException(status_code=503, detail=(
        f"Model artifact '{name}' is not available. Run 'python ml/train_supervised.py' and "
        "'python ml/analyze_unsupervised.py' to create it."))


@lru_cache
def load_model() -> dict:
    path = ARTIFACTS / "preference_model.joblib"
    if not path.exists():
        raise _missing(path.name)
    return joblib.load(path)


@lru_cache
def load_json(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        raise _missing(name)
    return json.loads(path.read_text())


@lru_cache
def load_dramas() -> pd.DataFrame:
    if not DRAMAS_CSV.exists():
        raise _missing(DRAMAS_CSV.name)
    return pd.read_csv(DRAMAS_CSV)


# --------------------------------------------------------------------------- schemas
class PredictionInput(BaseModel):
    """The inputs of the trained model. popularity_rank is converted to log10 exactly as in training."""
    genre_score: float | None = Field(
        default=None, ge=1, le=10,
        description="Viewer's usual score for this genre (1-10). Leave empty if unknown: the "
                    "pipeline then fills in the training median.")
    runtime_min: float = Field(ge=1, le=180, description="Minutes per episode")
    popularity_rank: float = Field(ge=1, le=100000, description="MyDramaList popularity rank (1 = most popular)")
    public_rating: float = Field(ge=1, le=10, description="Public MyDramaList rating (0-10)")
    episodes: float = Field(ge=1, le=300, description="Number of episodes")
    year: int = Field(ge=1990, le=2035, description="Release year")


def to_model_frame(values: dict, features: list[str]) -> pd.DataFrame:
    row = dict(values)
    row["popularity_log"] = math.log10(row["popularity_rank"])
    row["genre_score"] = float("nan") if row.get("genre_score") is None else row["genre_score"]
    return pd.DataFrame([row])[features]


def confidence_note(probability: float, threshold: float, test_accuracy: float) -> str:
    distance = abs(probability - threshold)
    if distance < 0.10:
        return "Very uncertain: the probability is close to the decision threshold."
    if distance < 0.25:
        return "Moderately confident: the probability is clearly on one side of the threshold, but not far."
    return (f"Fairly confident for this model. Remember that it was wrong for about "
            f"{round((1 - test_accuracy) * 100)}% of the test reviews.")


# --------------------------------------------------------------------------- endpoints
@app.get("/api/health")
def health() -> dict:
    status = {
        "model": (ARTIFACTS / "preference_model.joblib").exists(),
        "supervised_results": (ARTIFACTS / "supervised_results.json").exists(),
        "unsupervised_results": (ARTIFACTS / "unsupervised_results.json").exists(),
        "drama_catalogue": DRAMAS_CSV.exists(),
    }
    return {"status": "ok" if all(status.values()) else "degraded", "artifacts": status}


@app.get("/api/model")
def model_info() -> dict:
    """Input schema and honest performance numbers of the trained model."""
    bundle = load_model()
    results = load_json("supervised_results.json")
    ranges = bundle["feature_ranges"]
    inputs = []
    for name, info in FEATURE_INFO.items():
        inputs.append({"name": name, **info, "train_min": ranges[name]["min"],
                       "train_max": ranges[name]["max"], "train_median": ranges[name]["median"]})
    return {
        "model_name": bundle["model_name"],
        "threshold": bundle["threshold"],
        "inputs": inputs,
        "dataset": results["dataset"],
        "split": results["split"],
        "cross_validation": results["cross_validation"],
        "test": results["test"],
        "test_other_model_reference_only": results["test_other_model_reference_only"],
        "roc_curve": results["roc_curve"],
        "threshold_study": results["threshold_study_out_of_fold_training"],
        "interpretation": results["interpretation"],
        "disclosure": DISCLOSURE,
    }


@app.post("/api/predict")
def predict(payload: PredictionInput) -> dict:
    """Run the saved scikit-learn pipeline on one new record."""
    bundle = load_model()
    pipeline, features, threshold = bundle["pipeline"], bundle["features"], bundle["threshold"]
    values = payload.model_dump()
    frame = to_model_frame(values, features)
    probability = float(pipeline.predict_proba(frame)[0, 1])

    # "What-if" explanation: replace ONE input by its typical (training median) value and see how
    # the probability moves. It describes the model's behaviour, not a cause in the real world.
    ranges = bundle["feature_ranges"]
    effects = []
    for name in FEATURE_INFO:
        if values[name] is None:
            continue
        changed = dict(values)
        changed[name] = ranges[name]["median"]
        base = float(pipeline.predict_proba(to_model_frame(changed, features))[0, 1])
        effects.append({"feature": name, "label": FEATURE_INFO[name]["label"], "value": values[name],
                        "typical_value": ranges[name]["median"],
                        "probability_with_typical_value": base,
                        "effect": probability - base})
    effects.sort(key=lambda e: -abs(e["effect"]))

    warnings = []
    if values["genre_score"] is None:
        warnings.append("No genre score given: the model used the training median, as it does for "
                        "viewers without history. The prediction is less personal.")
    for name in FEATURE_INFO:
        v = values[name]
        if v is not None and not (ranges[name]["min"] <= v <= ranges[name]["max"]):
            warnings.append(f"{FEATURE_INFO[name]['label']} = {v:g} is outside the training range "
                            f"({ranges[name]['min']:g} to {ranges[name]['max']:g}); the model is extrapolating.")

    return {
        "probability_like": probability,
        "predicted_class": int(probability >= threshold),
        "predicted_label": "Likely to like it" if probability >= threshold else "Likely not a favourite",
        "threshold": threshold,
        "model_name": bundle["model_name"],
        "confidence_note": confidence_note(probability, threshold,
                                           load_json("supervised_results.json")["test"]["accuracy"]),
        "feature_effects": effects,
        "warnings": warnings,
        "disclosure": DISCLOSURE,
    }


@app.get("/api/dramas")
def dramas(q: str = Query(default="", max_length=80), limit: int = Query(default=20, ge=1, le=100)) -> dict:
    """Search the real drama catalogue (sorted by popularity) to pre-fill the predictor form."""
    table = load_dramas()
    if q.strip():
        table = table[table["title"].str.contains(q.strip(), case=False, regex=False)]
    columns = ["title", "year", "genres", "runtime_min", "episodes", "popularity_rank", "public_rating"]
    return {"total": int(len(table)), "items": table[columns].head(limit).to_dict(orient="records")}


@app.get("/api/insights")
def insights() -> dict:
    """Everything the Drama Insights page shows: real outputs of ml/analyze_unsupervised.py."""
    return load_json("unsupervised_results.json")


@app.get("/api/summary")
def summary() -> dict:
    """Small set of headline numbers for the Home and About pages (all read from the result files)."""
    sup = load_json("supervised_results.json")
    uns = load_json("unsupervised_results.json")
    iso_main = next(r for r in uns["isolation_forest"]["summary"]
                    if r["contamination"] == uns["isolation_forest"]["main_contamination"])
    featured = sorted(uns["points"], key=lambda p: p["popularity_rank"])[:12]
    return {
        "supervised": {
            "reviews": sup["dataset"]["rows"], "viewers": sup["dataset"]["viewers"],
            "dramas": sup["dataset"]["dramas"], "positive_rate": sup["dataset"]["positive_rate"],
            "missing_share_genre_score": sup["dataset"]["missing_share_genre_score"],
            "train_rows": sup["split"]["train_rows"], "test_rows": sup["split"]["test_rows"],
            "selected_model": sup["selected_model"],
            "cv_f1": {name: m["f1_mean"] for name, m in sup["cross_validation"]["models"].items()},
            "baseline_f1": sup["cross_validation"]["baseline_always_like_f1_mean"],
            "test": sup["test"],
        },
        "unsupervised": {
            "dramas": uns["dataset"]["observations"], "features": uns["dataset"]["features"],
            "chosen_k": uns["kmeans"]["chosen_k"], "silhouette": uns["kmeans"]["chosen_silhouette"],
            "dbscan_clusters": uns["dbscan"]["clusters"], "dbscan_noise": uns["dbscan"]["noise"],
            "pca_retained_2d": uns["pca"]["retained_2d"],
            "iso_contamination": iso_main["contamination"], "iso_flagged": iso_main["flagged"],
            "iso_also_noise": iso_main["also_dbscan_noise"],
        },
        "featured": [{k: p[k] for k in ("title", "year", "genres", "public_rating", "popularity_rank",
                                         "runtime_min", "episodes", "kmeans")} for p in featured],
    }
