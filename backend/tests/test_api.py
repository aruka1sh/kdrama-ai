"""API tests. Run from the project root:  pytest backend/tests -q"""
import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app import main  # noqa: E402

client = TestClient(main.app)
GOOD = {"genre_score": 9.0, "runtime_min": 70, "popularity_rank": 150, "public_rating": 8.6,
        "episodes": 16, "year": 2022}


def test_health_reports_all_artifacts():
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert all(body["artifacts"].values())


def test_prediction_equals_the_saved_pipeline():
    """The API must return exactly what the joblib pipeline returns - no simulated numbers."""
    body = client.post("/api/predict", json=GOOD).json()
    bundle = joblib.load(ROOT / "artifacts" / "preference_model.joblib")
    row = dict(GOOD, popularity_log=math.log10(GOOD["popularity_rank"]))
    expected = bundle["pipeline"].predict_proba(pd.DataFrame([row])[bundle["features"]])[0, 1]
    assert body["probability_like"] == pytest.approx(expected)
    assert body["predicted_class"] == int(expected >= bundle["threshold"])
    assert body["threshold"] == 0.5


def test_prediction_matches_training_script_and_notebook():
    results = json.loads((ROOT / "artifacts" / "supervised_results.json").read_text())
    body = client.post("/api/predict", json=results["new_record"]["input"]).json()
    assert body["probability_like"] == pytest.approx(results["new_record"]["probability_like"])
    notebook = ROOT / "notebooks" / "p1_notebook_results.json"
    if notebook.exists():
        assert body["probability_like"] == pytest.approx(json.loads(notebook.read_text())["new_record_probability"])


def test_higher_genre_score_gives_higher_probability():
    low = client.post("/api/predict", json=dict(GOOD, genre_score=3)).json()["probability_like"]
    high = client.post("/api/predict", json=dict(GOOD, genre_score=9.5)).json()["probability_like"]
    assert high > low


def test_missing_genre_score_is_allowed_and_explained():
    body = client.post("/api/predict", json=dict(GOOD, genre_score=None)).json()
    assert 0 <= body["probability_like"] <= 1
    assert any("training median" in w for w in body["warnings"])


@pytest.mark.parametrize("bad", [
    dict(GOOD, genre_score=11), dict(GOOD, runtime_min=-5), dict(GOOD, popularity_rank=0),
    dict(GOOD, public_rating="great"), {k: v for k, v in GOOD.items() if k != "episodes"},
])
def test_invalid_input_is_rejected(bad):
    assert client.post("/api/predict", json=bad).status_code == 422


def test_out_of_range_input_gets_a_warning():
    body = client.post("/api/predict", json=dict(GOOD, year=1995)).json()
    assert any("outside the training range" in w for w in body["warnings"])


def test_model_info_has_schema_and_real_metrics():
    body = client.get("/api/model").json()
    results = json.loads((ROOT / "artifacts" / "supervised_results.json").read_text())
    assert [i["name"] for i in body["inputs"]] == list(GOOD)
    assert body["test"]["f1"] == results["test"]["f1"]
    assert body["model_name"] == results["selected_model"]


def test_drama_search():
    body = client.get("/api/dramas", params={"q": "goblin"}).json()
    assert any(item["title"] == "Goblin" for item in body["items"])
    assert client.get("/api/dramas", params={"limit": 5}).json()["items"][0]["popularity_rank"] == 1


def test_insights_are_the_analysis_file():
    body = client.get("/api/insights").json()
    assert len(body["points"]) == body["dataset"]["observations"] >= 100
    assert {"pca", "tsne", "umap", "kmeans", "dbscan", "iso"} <= set(body["points"][0])
    assert "liked" not in json.dumps(body["dataset"])          # no target in the unsupervised output


def test_missing_artifact_gives_honest_503(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ARTIFACTS", tmp_path)
    main.load_model.cache_clear()
    main.load_json.cache_clear()
    try:
        response = client.post("/api/predict", json=GOOD)
        assert response.status_code == 503
        assert "not available" in response.json()["detail"]
        assert client.get("/api/insights").status_code == 503
    finally:
        main.load_model.cache_clear()
        main.load_json.cache_clear()


def test_summary_matches_result_files():
    body = client.get("/api/summary").json()
    sup = json.loads((ROOT / "artifacts" / "supervised_results.json").read_text())
    uns = json.loads((ROOT / "artifacts" / "unsupervised_results.json").read_text())
    assert body["supervised"]["test"] == sup["test"]
    assert body["unsupervised"]["dbscan_noise"] == uns["dbscan"]["noise"]
    assert len(body["featured"]) == 12 and body["featured"][0]["popularity_rank"] == 1
