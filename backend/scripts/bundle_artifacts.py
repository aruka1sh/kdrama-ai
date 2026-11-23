"""Copy the files the API needs into backend/data/ml so the backend folder can be deployed alone.

Run after retraining:  python backend/scripts/bundle_artifacts.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "backend" / "data" / "ml"
FILES = [ROOT / "artifacts" / "preference_model.joblib", ROOT / "artifacts" / "supervised_results.json",
         ROOT / "artifacts" / "unsupervised_results.json", ROOT / "data" / "processed" / "dramas.csv"]

TARGET.mkdir(parents=True, exist_ok=True)
for source in FILES:
    shutil.copy2(source, TARGET / source.name)
    print("bundled", source.relative_to(ROOT))
