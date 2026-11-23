"""Project 1 - Supervised Learning: will a viewer like a K-Drama?

Run:  python ml/train_supervised.py
Creates:
  artifacts/preference_model.joblib   the selected scikit-learn Pipeline (used by the website API)
  artifacts/supervised_results.json   every number shown in the report, slides and website
  artifacts/figures/p1_*.png          the figures used in the report and slides

The workflow follows the Practice 8 worksheet:
DATA -> SPLIT -> PREPARE -> TRAIN -> VALIDATE -> TEST -> DECIDE
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kdrama_data import LIKE_THRESHOLD, load_tables

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
SEED = 42
FEATURES = ["genre_score", "runtime_min", "popularity_log", "public_rating", "episodes", "year"]
TARGET = "liked"
DECISION_THRESHOLD = 0.5

INK, ROSE, LAVENDER, BURGUNDY = "#1b1a20", "#c97b8e", "#9d8cc9", "#7a1f3d"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def add_model_features(table: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row feature engineering (no statistics are learned, so it is safe before the split)."""
    table = table.copy()
    table["popularity_log"] = np.log10(table["popularity_rank"])   # rank 1 -> 0, rank 1000 -> 3
    return table


def make_models() -> dict[str, Pipeline]:
    """Both pipelines: imputer -> scaler -> classifier. All steps are fitted on training data only."""
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=3,
                                                  random_state=SEED, n_jobs=-1)),
        ]),
    }


def test_metrics(y_true, proba, threshold=DECISION_THRESHOLD) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "accuracy": float((pred == y_true).mean()),
    }


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- DATA
    dramas, reviews = load_tables(ROOT / "data" / "raw")
    data = add_model_features(reviews)
    X, y, viewers = data[FEATURES], data[TARGET], data["viewer_id"]

    # ---------------------------------------------------------------- SPLIT (80 / 20)
    # One viewer can write many reviews. All reviews of a viewer must stay on the same side of the
    # split, otherwise the model would be tested on people it has already seen (leakage).
    # StratifiedGroupKFold with 5 folds gives one stratified 20% fold with whole viewers in it.
    # shuffle=False makes the split fully deterministic and identical in every scikit-learn version
    # (with scikit-learn 1.6 the shuffled variant gave a poorly stratified test fold: 51% liked, not 58%).
    outer = StratifiedGroupKFold(n_splits=5, shuffle=False)
    train_idx, test_idx = next(outer.split(X, y, groups=viewers))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train = viewers.iloc[train_idx]
    assert set(g_train).isdisjoint(set(viewers.iloc[test_idx]))

    # ---------------------------------------------------------------- TRAIN + VALIDATE (3-fold CV)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=False)
    folds = list(cv.split(X_train, y_train, groups=g_train))       # identical folds for every model
    models = make_models()
    cv_results = {}
    for name, model in models.items():
        f1 = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
        auc = cross_val_score(model, X_train, y_train, cv=folds, scoring="roc_auc")
        cv_results[name] = {"f1_folds": f1.tolist(), "f1_mean": float(f1.mean()), "f1_std": float(f1.std()),
                            "roc_auc_mean": float(auc.mean())}
    baseline = DummyClassifier(strategy="constant", constant=1)     # "everybody likes everything"
    baseline_f1 = cross_val_score(baseline, X_train, y_train, cv=folds, scoring="f1")

    # ---------------------------------------------------------------- SELECT (training data only)
    best_name = max(cv_results, key=lambda n: cv_results[n]["f1_mean"])
    best_model = models[best_name]

    # Threshold study on out-of-fold TRAINING predictions (the test set is still untouched).
    oof = cross_val_predict(best_model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    threshold_table = []
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        m = test_metrics(y_train.to_numpy(), oof, t)
        threshold_table.append({"threshold": t, "precision": m["precision"], "recall": m["recall"],
                                "f1": m["f1"], "fp": m["confusion_matrix"]["fp"],
                                "fn": m["confusion_matrix"]["fn"]})

    # ---------------------------------------------------------------- TEST (once)
    for model in models.values():
        model.fit(X_train, y_train)
    proba_test = best_model.predict_proba(X_test)[:, 1]
    final = test_metrics(y_test.to_numpy(), proba_test)
    other_name = next(n for n in models if n != best_name)
    other = test_metrics(y_test.to_numpy(), models[other_name].predict_proba(X_test)[:, 1])
    fpr, tpr, _ = roc_curve(y_test, proba_test)

    # ---------------------------------------------------------------- INTERPRET
    lr_coef = dict(zip(FEATURES, models["Logistic Regression"].named_steps["classifier"].coef_[0].tolist()))
    rf_imp = dict(zip(FEATURES, models["Random Forest"].named_steps["classifier"].feature_importances_.tolist()))
    perm = permutation_importance(best_model, X_test, y_test, scoring="roc_auc", n_repeats=20,
                                  random_state=SEED)
    perm_imp = {f: {"mean": float(m), "std": float(s)}
                for f, m, s in zip(FEATURES, perm.importances_mean, perm.importances_std)}

    # ---------------------------------------------------------------- NEW RECORD
    new_record = {"genre_score": 9.0, "runtime_min": 70.0, "popularity_rank": 150.0,
                  "public_rating": 8.6, "episodes": 16.0, "year": 2022}
    new_X = add_model_features(pd.DataFrame([new_record]))[FEATURES]
    new_proba = float(best_model.predict_proba(new_X)[0, 1])

    # ---------------------------------------------------------------- SAVE
    train_medians = X_train.median().to_dict()
    feature_ranges = {f: {"min": float(X_train[f].min()), "max": float(X_train[f].max()),
                          "median": float(train_medians[f])} for f in FEATURES}
    feature_ranges["popularity_rank"] = {
        "min": float(data["popularity_rank"].iloc[train_idx].min()),
        "max": float(data["popularity_rank"].iloc[train_idx].max()),
        "median": float(data["popularity_rank"].iloc[train_idx].median())}

    joblib.dump({"pipeline": best_model, "model_name": best_name, "features": FEATURES,
                 "threshold": DECISION_THRESHOLD, "train_medians": train_medians,
                 "feature_ranges": feature_ranges}, ARTIFACTS / "preference_model.joblib")

    results = {
        "dataset": {
            "rows": int(len(data)), "viewers": int(viewers.nunique()),
            "dramas": int(data["drama_id"].nunique()), "features": FEATURES, "target": TARGET,
            "like_threshold": LIKE_THRESHOLD,
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "positive_rate": float(y.mean()),
            "missing_counts": {f: int(X[f].isna().sum()) for f in FEATURES},
            "missing_share_genre_score": float(X["genre_score"].isna().mean()),
            "describe": X.describe().round(3).to_dict(),
        },
        "split": {"train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
                  "test_share": float(len(test_idx) / len(data)),
                  "train_viewers": int(g_train.nunique()),
                  "test_viewers": int(viewers.iloc[test_idx].nunique()),
                  "train_positive_rate": float(y_train.mean()),
                  "test_positive_rate": float(y_test.mean()),
                  "method": "StratifiedGroupKFold(n_splits=5, shuffle=False), first fold = test"},
        "cross_validation": {"strategy": "StratifiedGroupKFold(n_splits=3, shuffle=False), "
                                         "groups = viewer_id, scoring = F1",
                             "models": cv_results,
                             "baseline_always_like_f1_mean": float(baseline_f1.mean())},
        "selected_model": best_name,
        "test": final,
        "test_other_model_reference_only": {"model": other_name, **other},
        "roc_curve": {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()},
        "threshold_study_out_of_fold_training": threshold_table,
        "interpretation": {"logistic_coefficients_standardized": lr_coef,
                           "random_forest_importance": rf_imp,
                           "permutation_importance_test_auc_drop": perm_imp},
        "new_record": {"input": new_record, "probability_like": new_proba,
                       "predicted_class": int(new_proba >= DECISION_THRESHOLD),
                       "threshold": DECISION_THRESHOLD},
        "feature_ranges": feature_ranges,
        "seed": SEED,
    }
    (ARTIFACTS / "supervised_results.json").write_text(json.dumps(results, indent=2))

    make_figures(data, cv_results, best_name, final, fpr, tpr, lr_coef, rf_imp, perm_imp)

    print(f"rows={len(data)} train={len(train_idx)} test={len(test_idx)}")
    for name, r in cv_results.items():
        print(f"CV  {name:20s} F1 folds={np.round(r['f1_folds'], 3)} mean={r['f1_mean']:.3f} sd={r['f1_std']:.3f}")
    print(f"CV  baseline (always like) F1 mean={baseline_f1.mean():.3f}")
    print("Selected:", best_name)
    print("Test:", json.dumps(final))
    print("New record probability:", round(new_proba, 3))


def make_figures(data, cv_results, best_name, final, fpr, tpr, lr_coef, rf_imp, perm_imp) -> None:
    liked = data[TARGET] == 1

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, f in zip(axes.ravel(), FEATURES):
        bins = np.histogram_bin_edges(data[f].dropna(), bins=25)
        ax.hist(data.loc[~liked, f].dropna(), bins=bins, alpha=0.65, color=LAVENDER, label="not liked (0)")
        ax.hist(data.loc[liked, f].dropna(), bins=bins, alpha=0.55, color=BURGUNDY, label="liked (1)")
        ax.set_title(f)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Feature distributions by class (real MyDramaList reviews)")
    fig.tight_layout()
    fig.savefig(FIGURES / "p1_feature_distributions.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    counts = data[TARGET].value_counts().sort_index()
    axes[0].bar(["not liked (0)", "liked (1)"], counts.values, color=[LAVENDER, BURGUNDY])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}\n{v / counts.sum():.1%}", ha="center", va="bottom")
    axes[0].set_ylim(0, counts.max() * 1.2)
    axes[0].set_title("Class balance")
    axes[1].hist(data["viewer_score"], bins=np.arange(0.75, 10.75, 0.5), color=ROSE)
    axes[1].axvline(LIKE_THRESHOLD - 0.25, color=INK, ls="--")
    axes[1].set_title("Viewer overall score (liked = score >= 8)")
    fig.tight_layout()
    fig.savefig(FIGURES / "p1_class_balance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    corr = data[FEATURES + [TARGET]].corr()
    im = ax.imshow(corr, cmap="RdPu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGURES / "p1_correlation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    names = list(cv_results)
    ax.bar(names, [cv_results[n]["f1_mean"] for n in names],
           yerr=[cv_results[n]["f1_std"] for n in names], capsize=6, color=[ROSE, LAVENDER])
    for i, n in enumerate(names):
        ax.text(i, cv_results[n]["f1_mean"] + 0.03, f"{cv_results[n]['f1_mean']:.3f}", ha="center")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Training-only 3-fold CV F1")
    ax.set_title("Model comparison (same folds)")
    fig.tight_layout()
    fig.savefig(FIGURES / "p1_cv_comparison.png")
    plt.close(fig)

    cm = final["confusion_matrix"]
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].imshow(matrix, cmap="RdPu")
    axes[0].grid(False)
    axes[0].set_xticks([0, 1], ["predicted 0", "predicted 1"])
    axes[0].set_yticks([0, 1], ["actual 0", "actual 1"])
    labels = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{labels[i][j]}\n{matrix[i, j]}", ha="center", va="center",
                         color="white" if matrix[i, j] > matrix.max() / 2 else INK, fontsize=12)
    axes[0].set_title(f"{best_name} - held-out test set")
    axes[1].plot(fpr, tpr, color=BURGUNDY, lw=2, label=f"ROC-AUC = {final['roc_auc']:.3f}")
    axes[1].plot([0, 1], [0, 1], color="gray", ls="--", label="random guess")
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate (recall)")
    axes[1].set_title("ROC curve - test set")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "p1_confusion_roc.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    pd.Series(lr_coef).sort_values().plot(kind="barh", ax=axes[0], color=ROSE)
    axes[0].axvline(0, color="gray")
    axes[0].set_title("Logistic Regression\nstandardized coefficients")
    pd.Series(rf_imp).sort_values().plot(kind="barh", ax=axes[1], color=LAVENDER)
    axes[1].set_title("Random Forest\nimpurity-based importance")
    perm_series = pd.Series({k: v["mean"] for k, v in perm_imp.items()}).sort_values()
    perm_series.plot(kind="barh", ax=axes[2], color=BURGUNDY,
                     xerr=[perm_imp[k]["std"] for k in perm_series.index])
    axes[2].set_title(f"{best_name}\npermutation importance (test AUC drop)")
    fig.tight_layout()
    fig.savefig(FIGURES / "p1_feature_importance.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
