"""Quality check: do the notebooks, the result files, the saved model and the documents agree?

Run after executing both notebooks:  python qa/check_consistency.py
Exits with an error message if anything does not match.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
NB = ROOT / "notebooks"
checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


S = json.loads((ART / "supervised_results.json").read_text())
U = json.loads((ART / "unsupervised_results.json").read_text())
n1 = json.loads((NB / "p1_notebook_results.json").read_text())
n2 = json.loads((NB / "p2_notebook_results.json").read_text())

# ---------------------------------------------------------------- notebooks executed without errors
for name in ["Project1_Supervised_KDrama.ipynb", "Project2_Unsupervised_KDrama.ipynb"]:
    nb = nbformat.read(NB / name, as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    errors = [o for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
    images = sum(1 for c in code for o in c.get("outputs", []) if "image/png" in o.get("data", {}))
    check(f"{name}: every code cell executed", all(c.get("execution_count") for c in code), f"{len(code)} code cells")
    check(f"{name}: no error outputs", not errors)
    check(f"{name}: figures present", images >= 6, f"{images} figures")
    text = "\n".join(c.source for c in nb.cells)
    if "Project2" in name:
        for algo in ["KMeans", "DBSCAN", "PCA(", "TSNE", "umap.UMAP", "IsolationForest"]:
            check(f"Project 2 notebook uses {algo}", algo in text)
        fit_lines = [line for line in text.splitlines() if ".fit" in line]
        check("Project 2: 'liked' is never used in a model fit", not any("liked" in line for line in fit_lines))
        check("Project 2: review table (with the target) is deleted", "del reviews" in text)
    else:
        for algo in ["LogisticRegression", "RandomForestClassifier", "StratifiedGroupKFold(n_splits=3", "Pipeline(["]:
            check(f"Project 1 notebook uses {algo}", algo in text)

# ---------------------------------------------------------------- Project 1: notebook == training script
check("P1 rows", n1["rows"] == S["dataset"]["rows"], str(n1["rows"]))
check("P1 train/test sizes", (n1["train_rows"], n1["test_rows"]) == (S["split"]["train_rows"], S["split"]["test_rows"]))
check("P1 test share is 80/20", abs(S["split"]["test_share"] - 0.20) < 0.01, f"{S['split']['test_share']:.4f}")
check("P1 selected model", n1["selected_model"] == S["selected_model"], n1["selected_model"])
for model, value in n1["cv_f1_mean"].items():
    check(f"P1 CV F1 {model}", close(value, S["cross_validation"]["models"][model]["f1_mean"]), f"{value:.6f}")
for metric, value in n1["test"].items():
    check(f"P1 test {metric}", close(value, S["test"][metric]), f"{value:.6f}")
check("P1 confusion matrix", n1["confusion_matrix"] == S["test"]["confusion_matrix"], str(n1["confusion_matrix"]))
best = S["selected_model"]
check("P1 selection follows the rule (highest mean CV F1)",
      best == max(S["cross_validation"]["models"], key=lambda m: S["cross_validation"]["models"][m]["f1_mean"]))

# metrics recomputed from the confusion matrix
cm = S["test"]["confusion_matrix"]
precision = cm["tp"] / (cm["tp"] + cm["fp"])
recall = cm["tp"] / (cm["tp"] + cm["fn"])
check("P1 precision = TP / (TP + FP)", close(precision, S["test"]["precision"]), f"{precision:.6f}")
check("P1 recall = TP / (TP + FN)", close(recall, S["test"]["recall"]), f"{recall:.6f}")
check("P1 F1 = harmonic mean", close(2 * precision * recall / (precision + recall), S["test"]["f1"]))
check("P1 confusion matrix sums to the test size", sum(cm.values()) == S["split"]["test_rows"])

# saved model reproduces the reported new-record probability
bundle = joblib.load(ART / "preference_model.joblib")
row = dict(S["new_record"]["input"])
row["popularity_log"] = math.log10(row["popularity_rank"])
proba = float(bundle["pipeline"].predict_proba(pd.DataFrame([row])[bundle["features"]])[0, 1])
check("P1 saved model == script == notebook (new record)",
      close(proba, S["new_record"]["probability_like"]) and close(proba, n1["new_record_probability"]), f"{proba:.6f}")
check("P1 pipeline steps are imputer -> scaler -> classifier",
      list(bundle["pipeline"].named_steps) == ["imputer", "scaler", "classifier"])
check("P1 target is not among the features",
      not {"liked", "viewer_score"} & set(bundle["features"]), ", ".join(bundle["features"]))

# ---------------------------------------------------------------- Project 2: notebook == analysis script
check("P2 observations >= 100", n2["observations"] == U["dataset"]["observations"] >= 100, str(n2["observations"]))
check("P2 at least three numerical features", len(U["dataset"]["features"]) >= 3, ", ".join(U["dataset"]["features"]))
for row_k in U["kmeans"]["k_curve"]:
    check(f"P2 silhouette K={row_k['k']}", close(n2["k_silhouette"][str(row_k["k"])], row_k["silhouette"]), f"{row_k['silhouette']:.6f}")
check("P2 K-Means sizes", n2["kmeans_sizes"] == [p["size"] for p in U["kmeans"]["profiles"]], str(n2["kmeans_sizes"]))
check("P2 DBSCAN clusters and noise", (n2["dbscan_clusters"], n2["dbscan_noise"]) == (U["dbscan"]["clusters"], U["dbscan"]["noise"]),
      f"{n2['dbscan_clusters']} clusters, {n2['dbscan_noise']} noise")
check("P2 PCA ratios", all(close(a, b) for a, b in zip(n2["pca_ratio"], U["pca"]["explained_variance_ratio"])))
for r in U["isolation_forest"]["summary"]:
    key = str(r["contamination"])
    check(f"P2 Isolation Forest c={key}", n2["iso_flagged"][key] == r["flagged"] and n2["iso_overlap_noise"][key] == r["also_dbscan_noise"],
          f"{r['flagged']} flagged, {r['also_dbscan_noise']} also noise")
check("P2 DBSCAN grid has several eps and min_samples",
      len({g["eps"] for g in U["dbscan"]["grid"]}) >= 2 and len({g["min_samples"] for g in U["dbscan"]["grid"]}) >= 2)
check("P2 at least two contamination values", len(U["isolation_forest"]["summary"]) >= 2)
check("P2 results contain no target", "liked" not in json.dumps(U))

# ---------------------------------------------------------------- deliverables exist
for path in ["reports/Report1_Supervised_Learning.docx", "reports/Report2_Unsupervised_Learning.docx",
             "presentation/KDrama_AI_Presentation.pptx", "docs/PRESENTATION_SCRIPT.md", "docs/DEFENSE_QA.md",
             "docs/REQUIREMENTS_CHECKLIST.md", "docs/TEAM_ROLES.md", "README.md"]:
    check(f"exists: {path}", (ROOT / path).exists())

# key numbers appear in the generated documents
from docx import Document  # noqa: E402

report1 = "\n".join(p.text for p in Document(ROOT / "reports" / "Report1_Supervised_Learning.docx").paragraphs)
report2 = "\n".join(p.text for p in Document(ROOT / "reports" / "Report2_Unsupervised_Learning.docx").paragraphs)
check("Report 1 quotes the real test F1", f"{S['test']['f1']:.3f}" in report1, f"{S['test']['f1']:.3f}")
check("Report 1 quotes the real ROC-AUC", f"{S['test']['roc_auc']:.3f}" in report1, f"{S['test']['roc_auc']:.3f}")
check("Report 2 quotes the real noise count", f"{U['dbscan']['noise']} noise points" in report2)
check("Report 2 quotes the real PCA variance", f"{U['pca']['retained_2d'] * 100:.1f}%" in report2)

failed = [c for c in checks if not c[1]]
for name, ok, detail in checks:
    print(("PASS  " if ok else "FAIL  ") + name + (f"  [{detail}]" if detail else ""))
print(f"\n{len(checks) - len(failed)} of {len(checks)} checks passed")
sys.exit(1 if failed else 0)
