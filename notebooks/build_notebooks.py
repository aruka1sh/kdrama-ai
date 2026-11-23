"""Builds the two Colab notebooks from plain Python strings.

Run:  python notebooks/build_notebooks.py        (then execute them, see README)

Why a builder? The data-loading code lives in ml/kdrama_data.py. The builder copies that code into
the first cells of each notebook, so the notebooks stay self-contained for Google Colab while the
website backend and the notebooks always use exactly the same data logic.
The numbers quoted in the written conclusions are read from artifacts/*.json (real outputs of
ml/train_supervised.py and ml/analyze_unsupervised.py) - nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "artifacts" / "supervised_results.json").read_text())
U = json.loads((ROOT / "artifacts" / "unsupervised_results.json").read_text())

data_source = (ROOT / "ml" / "kdrama_data.py").read_text()
data_source = data_source.split('if __name__ == "__main__":')[0].replace("from __future__ import annotations\n", "")
DATA_CELL = data_source.strip() + '\n\ndramas, reviews = load_tables("data/raw")\nprint("drama table :", dramas.shape)\nprint("review table:", reviews.shape)'

TEAM = "Team (QAIRU): Shakenova Aruzhan, Qabir Madina, Shahap Aizere, Zholdaspek Dana"


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}
    for kind, text in cells:
        text = text.strip("\n")
        nb.cells.append(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text))
    return nb


def md(text):
    return ("md", text)


def code(text):
    return ("code", text)


# =====================================================================================
# PROJECT 1 - SUPERVISED LEARNING
# =====================================================================================
cv = S["cross_validation"]["models"]
t = S["test"]
cm = t["confusion_matrix"]
ds = S["dataset"]
sp = S["split"]
perm = S["interpretation"]["permutation_importance_test_auc_drop"]
top_perm = sorted(perm, key=lambda f: -perm[f]["mean"])
other = S["test_other_model_reference_only"]

p1 = [
md(f"""
# K-Drama AI - Project 1: Supervised Learning
## Will a viewer like this K-Drama?

**Course:** AI Fundamentals - Final Project 1 (Practice 8 worksheet) &nbsp;|&nbsp; **Variant 21 - Movie preference**, adapted to Korean dramas

**{TEAM}**

| Item | Our choice |
|---|---|
| Object (one row) | one real review: one viewer scores one K-Drama on MyDramaList |
| Target | `liked` = 1 if the viewer's own overall score is 8 or more (out of 10), else 0 |
| Type | binary classification |
| Variant features | `genre_score`, `runtime_min`, `popularity` |
| Extra features | `public_rating`, `episodes`, `year` |
| Models | Logistic Regression and Random Forest (scikit-learn Pipelines) |

**How we adapted the variant.** Variant 21 says: *predict "likes movie" from genre score, runtime and popularity; no exact
source established*. We replaced movies with Korean dramas and found a **real** public dataset with genuine individual
viewer scores, so **no label in this notebook is simulated**. The original 1-10 score is grouped into two classes
(documented below). This adaptation of the task should be confirmed with the instructor, as the worksheet asks.

The notebook follows the worksheet sequence **DATA -> SPLIT -> PREPARE -> TRAIN -> VALIDATE -> TEST -> DECIDE**.
"""),
md("""
## Step 1 - Import the libraries

`pandas` and `numpy` hold the data, `matplotlib` draws the charts, and `scikit-learn` gives us the split,
the leakage-safe `Pipeline`, both classifiers and the metrics. Nothing is trained in this cell.
"""),
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (classification_report, confusion_matrix, ConfusionMatrixDisplay,
                             precision_score, recall_score, f1_score, roc_auc_score, roc_curve)

SEED = 42
pd.set_option("display.width", 160)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})
print("Libraries are ready.")
"""),
md("""
## Step 2 - Download the REAL data and build the table

**Sources (both scraped from MyDramaList.com and published on Kaggle):**

* **Source A** - Chanon Charuchinda (2023). *Korean Drama from 2015-2023 with Actors & Reviews*. Kaggle. Licence: CC0 Public Domain.
  <https://www.kaggle.com/datasets/chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist>
  We use `korean_drama.csv` (drama metadata) and `reviews.csv` (10,625 reviews written by real viewers, each with an `overall_score`).
* **Source B** - Noor Rizki (2023). *Top Korean Drama List (~1500)*. Kaggle. Licence: not stated ("Unknown").
  <https://www.kaggle.com/datasets/noorrizki/top-korean-drama-list-1500>
  We use `kdrama_list.csv` for the genres and the public MyDramaList score of each drama.

The cell below downloads the files directly (no Kaggle account is needed), joins them by title and year, and builds two tables:
`dramas` (one row per drama) and `reviews` (one row per viewer review). **Nothing is generated randomly.**

**The most important feature - `genre_score`.** The variant asks for a "genre score". We define it for each review as
*the average score that the same viewer gave to OTHER dramas that share at least one genre with this drama*.
The review we want to predict is never included in this average, so the feature does not contain the answer.
If the viewer has no other review with a shared genre, the value is missing (NaN). These are real missing values.
"""),
code(DATA_CELL),
md("""
## Step 3 - Inspect the data

We look at the size, the first rows, the data types, the missing values and the class balance.
"""),
code("""
MODEL_COLUMNS = ["genre_score", "runtime_min", "popularity_rank", "public_rating", "episodes", "year"]

print("Dataset dimensions (rows, columns):", reviews.shape)
print("Different viewers:", reviews["viewer_id"].nunique(), "| different dramas:", reviews["drama_id"].nunique())
display(reviews.drop(columns=["viewer_id", "drama_id"]).head())

feature_info = pd.DataFrame({
    "column": MODEL_COLUMNS + ["liked"],
    "dtype": [str(reviews[c].dtype) for c in MODEL_COLUMNS + ["liked"]],
    "meaning": [
        "viewer's mean score (1-10) for OTHER dramas sharing a genre with this one; NaN = no such history",
        "length of one episode in minutes",
        "MyDramaList popularity rank (1 = most popular drama)",
        "public MyDramaList rating of the drama (average of all users, 0-10)",
        "number of episodes",
        "release year",
        "TARGET: 1 if this viewer's overall score >= 8, else 0",
    ],
})
display(feature_info)
"""),
code("""
print("Missing values per column:")
print(reviews[MODEL_COLUMNS + ["liked"]].isna().sum().to_string())
print("\\nShare of missing genre_score:", round(reviews["genre_score"].isna().mean(), 3))

print("\\nClass counts:      ", reviews["liked"].value_counts().sort_index().to_dict())
print("Class proportions: ", reviews["liked"].value_counts(normalize=True).sort_index().round(3).to_dict())
display(reviews[MODEL_COLUMNS].describe().round(2))
"""),
md(f"""
**What the result means**

* The table has **{ds['rows']:,} real reviews** written by **{ds['viewers']:,} viewers** about **{ds['dramas']:,} dramas**.
* All model columns are numerical. Only `genre_score` has missing values: **{ds['missing_counts']['genre_score']:,} rows
  ({ds['missing_share_genre_score']:.1%})**. They belong to viewers with no other review of a similar genre (mostly viewers with one review).
  We did **not** add artificial missing values - the worksheet's worked example says missingness should reflect the actual dataset. The gaps are filled later *inside* the pipeline.
* Class balance: **{ds['positive_rate']:.1%} liked (1)** and **{1 - ds['positive_rate']:.1%} not liked (0)**. The classes are a little unbalanced but not extreme,
  so F1 and ROC-AUC are more informative than accuracy alone.
* **Binary grouping:** the source has scores from 1 to 10 in steps of 0.5. We code 8.0-10 as class 1 ("liked") and 1-7.5 as class 0. The median score is 8, so this threshold splits viewers into "clearly positive" and "the rest".
* **Limitation:** people who write reviews are not typical viewers. They choose what to watch and review, so the data is not a random sample.
"""),
md("""
## Step 4 - Exploratory Data Analysis (four figures)
"""),
code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
counts = reviews["liked"].value_counts().sort_index()
axes[0].bar(["not liked (0)", "liked (1)"], counts.values, color=["#9d8cc9", "#7a1f3d"])
axes[0].set_title("Figure 1a - class balance")
axes[1].hist(reviews["viewer_score"], bins=np.arange(0.75, 10.75, 0.5), color="#c97b8e")
axes[1].axvline(7.75, color="black", linestyle="--")
axes[1].set_title("Figure 1b - viewer overall score (liked = 8 or more)")
plt.tight_layout(); plt.show()
"""),
md("""
**Figure 1.** Reviewers are generous: most scores are 7 or higher and the most common values are 8-10.
The dashed line is our threshold. A model that always answers "liked" would already be right more than half of the time,
so we must compare our models with this simple baseline.
"""),
code("""
liked = reviews["liked"] == 1
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for ax, col in zip(axes.ravel(), MODEL_COLUMNS):
    bins = np.histogram_bin_edges(reviews[col].dropna(), bins=25)
    ax.hist(reviews.loc[~liked, col].dropna(), bins=bins, alpha=0.65, color="#9d8cc9", label="not liked (0)")
    ax.hist(reviews.loc[liked, col].dropna(), bins=bins, alpha=0.55, color="#7a1f3d", label="liked (1)")
    ax.set_title(col)
axes[0, 0].legend()
fig.suptitle("Figure 2 - feature distributions by class")
plt.tight_layout(); plt.show()
"""),
md("""
**Figure 2.** (1) `genre_score`: liked reviews are shifted to the right - viewers who usually enjoy a genre tend to like the next drama of that genre.
(2) `public_rating`: liked reviews are more frequent for highly rated dramas. (3) `runtime_min` has two peaks: short web dramas (10-25 minutes)
and normal TV dramas (60-70 minutes). (4) `popularity_rank` and `episodes` are strongly right-skewed, and the features have very different scales
(years are about 2,000, ranks go up to 10,000) - this is why we scale them for Logistic Regression.
"""),
code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
genre_bins = pd.cut(reviews["genre_score"], bins=[0, 5, 6, 7, 8, 9, 10])
reviews.groupby(genre_bins, observed=True)["liked"].mean().plot(kind="bar", ax=axes[0], color="#c97b8e")
axes[0].set_title("Figure 3a - share of liked reviews by genre_score")
axes[0].set_ylabel("share liked"); axes[0].set_ylim(0, 1)
rating_bins = pd.cut(reviews["public_rating"], bins=[6, 7, 7.5, 8, 8.5, 9.5])
reviews.groupby(rating_bins, observed=True)["liked"].mean().plot(kind="bar", ax=axes[1], color="#9d8cc9")
axes[1].set_title("Figure 3b - share of liked reviews by public_rating")
axes[1].set_ylim(0, 1)
plt.tight_layout(); plt.show()
"""),
md("""
**Figure 3.** The share of "liked" grows step by step with both `genre_score` and `public_rating`.
The relationship is clear but not perfect: even in the best bins some viewers do not like the drama. We should expect a useful model, not a perfect one.
"""),
code("""
corr = reviews[MODEL_COLUMNS + ["liked"]].corr()
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr, cmap="RdPu", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
ax.grid(False); fig.colorbar(im, shrink=0.8)
ax.set_title("Figure 4 - correlation matrix")
plt.tight_layout(); plt.show()
"""),
md("""
**Figure 4.** `genre_score` and `public_rating` have the strongest correlation with the target. `public_rating` and `popularity_rank`
are negatively correlated (popular dramas have a small rank number and a higher rating). Correlation shows association only, not cause.
"""),
md("""
## Step 5 - Feature engineering and the 80/20 split (before any fitted preprocessing)

**Feature engineering.** `popularity_rank` is very skewed, so we use `popularity_log = log10(popularity_rank)`.
This is a row-by-row formula: it learns nothing from other rows, so it is safe to apply before the split.

**Why a *grouped* split?** One viewer can write many reviews, and `genre_score` is built from the viewer's other reviews.
If the same viewer were in both the training and the test set, the test would be too easy (a hidden form of data leakage).
So we keep **all reviews of one viewer on the same side**. `StratifiedGroupKFold` with 5 folds creates five parts with
(almost) the same class balance and no shared viewers; we take one part (20%) as the test set and the other four (80%) for training.
A plain `train_test_split(stratify=y)` cannot keep the viewers together.

**Why `shuffle=False`?** Without shuffling the procedure has no randomness at all, so everybody gets exactly the same split,
in every version of scikit-learn. (We checked: with scikit-learn 1.6 the shuffled variant returned a poorly stratified test fold -
51% liked instead of 58% - while the non-shuffled variant is identical in versions 1.6 and 1.9.)
"""),
code("""
data = reviews.copy()
data["popularity_log"] = np.log10(data["popularity_rank"])

FEATURES = ["genre_score", "runtime_min", "popularity_log", "public_rating", "episodes", "year"]
X = data[FEATURES]
y = data["liked"]
viewers = data["viewer_id"]

outer = StratifiedGroupKFold(n_splits=5, shuffle=False)
train_idx, test_idx = next(outer.split(X, y, groups=viewers))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
g_train, g_test = viewers.iloc[train_idx], viewers.iloc[test_idx]

print("Train:", X_train.shape, "| Test:", X_test.shape, "| test share:", round(len(test_idx) / len(data), 3))
print("Liked share - train:", round(y_train.mean(), 3), "| test:", round(y_test.mean(), 3))
print("Viewers in both sets:", len(set(g_train) & set(g_test)))
"""),
md(f"""
**What the result means.** {sp['train_rows']:,} rows are used for training and {sp['test_rows']:,} rows ({sp['test_share']:.1%}) are locked away for the final test.
The class balance is almost identical in both parts ({sp['train_positive_rate']:.1%} and {sp['test_positive_rate']:.1%}), and no viewer appears in both.
The imputer and the scaler have **not** been fitted yet - they will learn their numbers from training rows only.
"""),
md("""
## Step 6 - Build two leakage-safe Pipelines

Each pipeline has three steps: **median imputer -> standard scaler -> classifier**. Because the steps are inside a `Pipeline`,
the median and the scaling numbers are always learned from the training part only (also inside every cross-validation fold).

* **Logistic Regression** - a simple linear model. It gives probabilities and coefficients that are easy to explain. It is our transparent baseline model.
* **Random Forest** - many decision trees that vote. It can learn non-linear rules (for example "high genre score *and* high rating").
  Trees do not need scaling; the scaler is kept only to mirror the worksheet workflow.
"""),
code("""
lr_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=SEED)),
])
rf_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=3,
                                          random_state=SEED, n_jobs=-1)),
])
models = {"Logistic Regression": lr_model, "Random Forest": rf_model}
print(lr_model)
"""),
md("""
## Step 7 - Validate both models with the same stratified 3-fold cross-validation

We create the three folds **once** and give the same folds to both models, so the comparison is fair.
The folds are stratified (same class balance), grouped by viewer and deterministic (`shuffle=False`, as explained above). The test set is not used here.

For each model we also compute the confusion matrix, precision, recall, F1 and ROC-AUC from **out-of-fold** predictions:
every training row is predicted by a model that did not see it.
"""),
code("""
cv = StratifiedGroupKFold(n_splits=3, shuffle=False)
folds = list(cv.split(X_train, y_train, groups=g_train))

cv_f1, rows = {}, []
for name, model in models.items():
    cv_f1[name] = cross_val_score(model, X_train, y_train, cv=folds, scoring="f1")
    oof_proba = cross_val_predict(model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
    oof_label = (oof_proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_train, oof_label, labels=[0, 1]).ravel()
    rows.append({"model": name, "F1 fold 1": cv_f1[name][0], "F1 fold 2": cv_f1[name][1], "F1 fold 3": cv_f1[name][2],
                 "mean F1": cv_f1[name].mean(), "SD": cv_f1[name].std(),
                 "precision": precision_score(y_train, oof_label), "recall": recall_score(y_train, oof_label),
                 "ROC-AUC": roc_auc_score(y_train, oof_proba), "TN": tn, "FP": fp, "FN": fn, "TP": tp})

baseline = DummyClassifier(strategy="constant", constant=1)      # always answers "liked"
baseline_f1 = cross_val_score(baseline, X_train, y_train, cv=folds, scoring="f1")

cv_table = pd.DataFrame(rows).set_index("model")
display(cv_table.round(3))
print("Baseline (always 'liked') mean F1:", round(baseline_f1.mean(), 3))

plt.figure(figsize=(5.5, 3.6))
plt.bar(list(cv_f1), [v.mean() for v in cv_f1.values()], yerr=[v.std() for v in cv_f1.values()],
        capsize=6, color=["#c97b8e", "#9d8cc9"])
plt.axhline(baseline_f1.mean(), color="gray", linestyle="--", label="always 'liked' baseline")
plt.ylim(0, 1); plt.ylabel("Training-only CV F1"); plt.title("Figure 5 - model comparison (same folds)"); plt.legend()
plt.show()
"""),
md(f"""
**What the result means.**

| Model | mean CV F1 | SD |
|---|---|---|
| Logistic Regression | {cv['Logistic Regression']['f1_mean']:.3f} | {cv['Logistic Regression']['f1_std']:.3f} |
| Random Forest | {cv['Random Forest']['f1_mean']:.3f} | {cv['Random Forest']['f1_std']:.3f} |
| Baseline: always "liked" | {S['cross_validation']['baseline_always_like_f1_mean']:.3f} | - |

Both models are better than the baseline, but only by a few F1 points, and the two models are very close to each other
(the difference is about the size of one standard deviation). F1 hides part of the progress here because the positive class is the majority;
ROC-AUC (about {cv['Random Forest']['roc_auc_mean']:.2f} in cross-validation, where 0.5 is random guessing) shows that the models really rank viewers better than chance.
"""),
md("""
## Step 8 - Select the best model BEFORE looking at the test set

Rule from the worksheet: choose the model with the higher **mean cross-validation F1** on training data.
We do not choose by test results.
"""),
code("""
best_name = max(cv_f1, key=lambda name: cv_f1[name].mean())
best_model = models[best_name]
print("Selected model:", best_name, "| mean CV F1 =", round(cv_f1[best_name].mean(), 3))
"""),
md("""
## Step 9 - Final evaluation on the untouched test set

The selected pipeline is fitted on all training rows and evaluated **once** on the test rows.
"""),
code("""
THRESHOLD = 0.5
best_model.fit(X_train, y_train)
test_proba = best_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= THRESHOLD).astype(int)

print("Selected:", best_name)
print(classification_report(y_test, test_pred, target_names=["not liked (0)", "liked (1)"], zero_division=0))
cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
final = {"precision": precision_score(y_test, test_pred), "recall": recall_score(y_test, test_pred),
         "f1": f1_score(y_test, test_pred), "roc_auc": roc_auc_score(y_test, test_proba),
         "accuracy": (test_pred == y_test).mean()}
print("Confusion matrix [[TN FP] [FN TP]]:\\n", cm)
print({k: round(v, 3) for k, v in final.items()})

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ConfusionMatrixDisplay(cm, display_labels=["not liked (0)", "liked (1)"]).plot(ax=axes[0], cmap="RdPu", colorbar=False)
axes[0].grid(False); axes[0].set_title(f"Figure 6a - {best_name}, held-out test")
fpr, tpr, _ = roc_curve(y_test, test_proba)
axes[1].plot(fpr, tpr, color="#7a1f3d", label=f"ROC-AUC = {final['roc_auc']:.3f}")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", label="random guess")
axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate (recall)")
axes[1].set_title("Figure 6b - ROC curve, test set"); axes[1].legend()
plt.tight_layout(); plt.show()
"""),
md(f"""
**What the result means (test set, {sp['test_rows']:,} reviews, threshold 0.5).** Selected model: **{S['selected_model']}**.

| Metric | Value | Simple meaning |
|---|---|---|
| Confusion matrix | TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']} | rows = real class, columns = predicted class |
| Precision | {t['precision']:.3f} | when the model says "you will like it", it is right about {t['precision']:.0%} of the time |
| Recall | {t['recall']:.3f} | the model finds about {t['recall']:.0%} of the dramas that the viewer really liked |
| F1 | {t['f1']:.3f} | one number that balances precision and recall |
| ROC-AUC | {t['roc_auc']:.3f} | if we take one liked and one not-liked review, the model gives the liked one a higher probability in about {t['roc_auc']:.0%} of pairs (0.5 = guessing, 1.0 = perfect) |
| Accuracy | {t['accuracy']:.3f} | share of all correct answers (always answering "liked" would give {sp['test_positive_rate']:.3f}) |

* **False positives ({cm['fp']})** - the model promised "you will like it" but the viewer scored the drama below 8.
* **False negatives ({cm['fn']})** - the model said "not for you" but the viewer actually liked it.
* The test F1 ({t['f1']:.3f}) is close to the cross-validation F1 ({cv[S['selected_model']]['f1_mean']:.3f}), so the model does not look over-fitted.

For reference only (not used for the choice): {other['model']} reaches F1 = {other['f1']:.3f} and ROC-AUC = {other['roc_auc']:.3f} on the same test set. The two models are practically equal.
"""),
md("""
## Step 10 - Which features influence the predictions?

We use three views:
1. **Logistic Regression coefficients** (standardized) - direction and strength in a linear model.
2. **Random Forest impurity importance** - how much each feature helps the trees to split.
3. **Permutation importance** of the selected model on the test set - how much ROC-AUC drops when one feature is shuffled.
"""),
code("""
for model in models.values():
    model.fit(X_train, y_train)

coefs = pd.Series(lr_model.named_steps["classifier"].coef_[0], index=FEATURES).sort_values()
importances = pd.Series(rf_model.named_steps["classifier"].feature_importances_, index=FEATURES).sort_values()
perm = permutation_importance(best_model, X_test, y_test, scoring="roc_auc", n_repeats=20, random_state=SEED)
perm_imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values()

fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
coefs.plot(kind="barh", ax=axes[0], color="#c97b8e"); axes[0].axvline(0, color="gray")
axes[0].set_title("Figure 7a - Logistic Regression coefficients")
importances.plot(kind="barh", ax=axes[1], color="#9d8cc9"); axes[1].set_title("Figure 7b - Random Forest importance")
perm_imp.plot(kind="barh", ax=axes[2], color="#7a1f3d"); axes[2].set_title(f"Figure 7c - permutation importance ({best_name})")
plt.tight_layout(); plt.show()

display(pd.DataFrame({"LR coefficient": coefs, "RF importance": importances, "AUC drop when shuffled": perm_imp}).round(3))
"""),
md(f"""
**What the result means.**

* All three views put **`genre_score`** and **`public_rating`** clearly on top (their order differs a little between the views), followed by `popularity_log`.
  Shuffling `{top_perm[0]}` lowers the test ROC-AUC by about {perm[top_perm[0]]['mean']:.2f}; shuffling `runtime_min` or `episodes` changes almost nothing.
* Positive coefficients (`genre_score`, `public_rating`) raise the predicted chance of "liked" when the other features stay the same.
* `popularity_log` has a positive coefficient: for the same public rating, reviewers liked less famous dramas slightly more often.
  One possible reason is self-selection (people who search for a niche drama already expect to like it), but our data cannot prove the reason.
* **Important:** importance shows what the *model uses*, not what *causes* a person to like a drama. Correlated features (rating and popularity) share importance between them.
"""),
md("""
## Step 11 - Predict one NEW record, decision threshold and the cost of errors

A new case: a viewer who usually gives **9.0** to dramas of this genre considers a drama with **70-minute** episodes,
popularity rank **150**, public rating **8.6**, **16** episodes, released in **2022**.
"""),
code("""
new_record = pd.DataFrame([{"genre_score": 9.0, "runtime_min": 70.0, "popularity_rank": 150.0,
                            "public_rating": 8.6, "episodes": 16.0, "year": 2022}])
new_record["popularity_log"] = np.log10(new_record["popularity_rank"])
new_proba = best_model.predict_proba(new_record[FEATURES])[0, 1]

print("Probability of class 1 (liked):", round(new_proba, 3))
print("Decision threshold:", THRESHOLD)
print("Predicted class:", int(new_proba >= THRESHOLD), "->", "LIKED" if new_proba >= THRESHOLD else "NOT LIKED")
"""),
code("""
# What happens if we move the threshold? (out-of-fold TRAINING predictions - the test set is not reused)
oof = cross_val_predict(best_model, X_train, y_train, cv=folds, method="predict_proba")[:, 1]
rows = []
for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
    label = (oof >= thr).astype(int)
    tn_, fp_, fn_, tp_ = confusion_matrix(y_train, label, labels=[0, 1]).ravel()
    rows.append({"threshold": thr, "precision": precision_score(y_train, label), "recall": recall_score(y_train, label),
                 "F1": f1_score(y_train, label), "false positives": fp_, "false negatives": fn_})
threshold_table = pd.DataFrame(rows)
display(threshold_table.round(3))
"""),
md(f"""
**What the result means.**

* The model gives the new record a probability of **{S['new_record']['probability_like']:.3f}**. It is above the threshold 0.5, so the predicted class is **1 (liked)**.
  This is a software demonstration with invented input values, not a validation on a real new viewer.
* **Decision threshold.** The model outputs a probability; the threshold turns it into a yes/no answer. A higher threshold makes the model more careful:
  fewer false positives (higher precision) but more false negatives (lower recall). The table above shows this trade-off.
* **Cost of errors** (for a recommendation service):
  * a **false positive** = we recommend a drama and the viewer does not enjoy it: wasted evening (a K-Drama is 16+ hours), less trust in recommendations;
  * a **false negative** = we hide a drama that the viewer would have loved: a missed opportunity that nobody notices.
* We keep **0.5** as the default because it gives a balanced F1. If wasted time is considered more expensive, a threshold of 0.6 would be a reasonable choice
  (precision goes up, recall goes down). Nobody is harmed by either error, so there is no need for an extreme threshold.
"""),
md(f"""
## Step 12 - Conclusions

**Summary.** We used {ds['rows']:,} real MyDramaList reviews ({ds['viewers']:,} viewers, {ds['dramas']:,} Korean dramas) from two public Kaggle datasets to predict whether a viewer scores a drama 8 or higher.
We split the data 80/20 by viewer, fitted the imputer and the scaler only inside pipelines, and compared Logistic Regression with Random Forest using the same stratified 3-fold cross-validation.
The model selected by mean validation F1 was **{S['selected_model']}** (mean CV F1 {cv[S['selected_model']]['f1_mean']:.3f}; the other model {cv[other['model']]['f1_mean']:.3f}).
On the held-out test set it reached **F1 = {t['f1']:.3f}, precision = {t['precision']:.3f}, recall = {t['recall']:.3f} and ROC-AUC = {t['roc_auc']:.3f}**, with {cm['fp']} false positives and {cm['fn']} false negatives.
The most influential features were `{top_perm[0]}` and `{top_perm[1]}`.

**Limitations**

1. **Self-selected reviewers.** People who write reviews are more active and more positive than ordinary viewers; results may not transfer to all viewers.
2. **Cold start.** For {ds['missing_share_genre_score']:.0%} of the rows the viewer has no history, so `genre_score` is filled with the training median. Predictions for new viewers are weaker.
3. **No time stamps.** `genre_score` uses the viewer's other reviews without knowing which was written first, and `public_rating` is a 2023 snapshot that also contains (as a tiny part) the reviewers' own scores. A real system must use only past information.
4. **The label is a simplification.** "Score of 8 or more" is our definition of "liked"; a different threshold would change the classes.
5. **Modest gain.** The models are only a few F1 points above the "always liked" baseline; ROC-AUC of about {t['roc_auc']:.2f} is useful but far from perfect.
6. **Licence.** Source B does not state a licence; we use it for a classroom project only.

**Risks.** A recommender trained on such data may push already popular, highly rated dramas and hide niche ones (popularity bias). Feature importance must not be read as causation.

**Realistic improvements.** (1) Add review dates and build `genre_score` only from earlier reviews; (2) add per-genre scores and tags (for example one score for Romance, one for Thriller);
(3) collect ratings from ordinary watch-lists, not only written reviews; (4) tune the threshold with a real cost estimate; (5) try gradient boosting and calibrate the probabilities.

**Reproducibility.** The split and the folds are deterministic (no shuffling); `SEED = 42` is used for both models. Run all cells from top to bottom in Google Colab; the data is downloaded automatically.

### Answers to the worksheet questions
* **Public labelled dataset vs synthetic practice dataset?** Here the labels were written by real people, so the results describe real behaviour (with its bias). Synthetic labels only describe our own assumptions.
* **Features and exact target?** Six numerical features (`genre_score`, `runtime_min`, `popularity_log`, `public_rating`, `episodes`, `year`); target `liked` = 1 if the viewer's overall score is 8 or more.
* **Why must the split come before preprocessing?** Otherwise the median and the scaling numbers would be calculated with test rows, and information from the test set would leak into training.
* **Which model has the higher CV F1?** {S['selected_model']} ({cv[S['selected_model']]['f1_mean']:.3f} against {cv[other['model']]['f1_mean']:.3f}).
* **Test false positives and false negatives?** FP = {cm['fp']}, FN = {cm['fn']}.
* **What does the importance plot tell us, and what not?** It tells which features the model relies on. It does not prove that these features cause people to like a drama.

### References
* Charuchinda, C. (2023). *Korean Drama from 2015-2023 with Actors & Reviews* [Data set]. Kaggle. CC0 Public Domain. https://www.kaggle.com/datasets/chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist
* Rizki, N. (2023). *Top Korean Drama List (~1500)* [Data set]. Kaggle. https://www.kaggle.com/datasets/noorrizki/top-korean-drama-list-1500
* MyDramaList. https://mydramalist.com (original source of both datasets)
* Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
"""),
code("""
# Key numbers of this run, saved so that the report and the website can be checked against the notebook.
import json
summary = {"rows": int(len(data)), "train_rows": int(len(train_idx)), "test_rows": int(len(test_idx)),
           "cv_f1_mean": {name: float(v.mean()) for name, v in cv_f1.items()},
           "selected_model": best_name, "test": {k: float(v) for k, v in final.items()},
           "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
           "new_record_probability": float(new_proba)}
with open("p1_notebook_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
"""),
]

# =====================================================================================
# PROJECT 2 - UNSUPERVISED LEARNING
# =====================================================================================
km = U["kmeans"]
db = U["dbscan"]
pca = U["pca"]
iso = U["isolation_forest"]
pre = U["preprocessing"]
kc = {r["k"]: r for r in km["k_curve"]}
iso_main = next(r for r in iso["summary"] if r["contamination"] == iso["main_contamination"])
grid = {(r["eps"], r["min_samples"]): r for r in db["grid"]}
profile_rows = "\n".join(
    f"| {p['cluster']} | {p['size']} ({p['share']:.0%}) | {p['mean']['runtime_min']:.0f} | {p['mean']['episodes']:.0f} | "
    f"{p['median']['popularity_rank']:.0f} | {p['mean']['public_rating']:.2f} | {p['mean']['genre_score']:.2f} | {p['description']} |"
    for p in km["profiles"])
iso_rows = "\n".join(
    f"| {r['contamination']} | {r['flagged']} | {r['also_dbscan_noise']} | {r['only_isolation_forest']} | {r['only_dbscan_noise']} |"
    for r in iso["summary"])
top_titles = ", ".join(f"*{a['title']}*" for a in iso["top_anomalies"][:5])
imp = pre["imputation_experiment_artificial"]["strategies"]
sc = pre["scaling_experiment"]

p2 = [
md(f"""
# K-Drama AI - Project 2: Unsupervised Learning
## Hidden patterns of K-Dramas: groups, low-dimensional structure and unusual dramas

**Course:** AI Fundamentals - Final Project (Unsupervised Learning worksheet) &nbsp;|&nbsp; **Variant 21 - Movie preference**, adapted to Korean dramas

**{TEAM}**

| Item | Our choice |
|---|---|
| Object | Korean TV dramas listed on MyDramaList (2015-2023) |
| Observation unit | one drama (one row = one drama) |
| Research question | Do K-Dramas form natural groups by genre score, runtime and popularity? Is there a simple low-dimensional structure? Which dramas are unusual? |
| Variant features | `genre_score`, `runtime_min`, `popularity` |
| Extra features | `episodes`, `public_rating` |
| Label | **none** - the "liked" target of Project 1 is not loaded and not used anywhere in this notebook |

Methods: K-Means, DBSCAN, PCA, t-SNE, UMAP and Isolation Forest. All figures and numbers are produced by the code below.
"""),
md("""
## Step 1 - Install and import the libraries

Google Colab already has NumPy, pandas, Matplotlib and scikit-learn. UMAP lives in the package `umap-learn`
(it is imported as `umap`), so we install it only if it is missing.
"""),
code("""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score

try:
    import umap
except ImportError:
    %pip -q install umap-learn
    import umap

SEED = 42
pd.set_option("display.width", 170)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25})
PALETTE = ["#c97b8e", "#9d8cc9", "#e0b973", "#5fa8a3", "#d98f5c", "#6f86c9", "#7a1f3d", "#b8b8c4"]
print("Libraries are ready.")
"""),
md("""
## Step 2 - Download the real data and inspect X

**Sources (both scraped from MyDramaList.com and published on Kaggle):**

* **Source A** - Chanon Charuchinda (2023). *Korean Drama from 2015-2023 with Actors & Reviews*. Kaggle, CC0 Public Domain.
  <https://www.kaggle.com/datasets/chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist> - runtime, episodes, popularity rank.
* **Source B** - Noor Rizki (2023). *Top Korean Drama List (~1500)*. Kaggle, licence not stated.
  <https://www.kaggle.com/datasets/noorrizki/top-korean-drama-list-1500> - genres and public rating.

The loader is the same code that Project 1 uses. It returns two tables; **this project uses only the `dramas` table**.
The `reviews` table (which contains the Project 1 target) is deleted immediately so it cannot be used by mistake.

**Drama-level `genre_score`.** First we compute the average public rating of every genre. Then, for each drama, we average
these values over the drama's genres. It answers: "how well-rated are this drama's genres in general?". No viewer label is involved.
"""),
code(DATA_CELL + '\ndel reviews            # the supervised target is not used in this project'),
code("""
RAW_FEATURES = ["genre_score", "runtime_min", "popularity_rank", "episodes", "public_rating"]

print("Observations and columns:", dramas.shape)
display(dramas.drop(columns=["drama_id"]).head())
feature_info = pd.DataFrame({
    "feature": RAW_FEATURES,
    "dtype": dramas[RAW_FEATURES].dtypes.astype(str).values,
    "meaning": ["average public rating of the drama's genres (0-10)",
                "length of one episode in minutes",
                "MyDramaList popularity rank (1 = most popular)",
                "number of episodes",
                "public MyDramaList rating of the drama (0-10)"],
})
display(feature_info)
print("Missing values:")
print(dramas[RAW_FEATURES].isna().sum().to_string())
display(dramas[RAW_FEATURES].describe().round(2))
"""),
md(f"""
**What the result means.** We have **{U['dataset']['observations']:,} dramas** (the requirement is at least 100) and five numerical features.
There are **no missing values** in these features. Reason: Source A has some gaps (24 dramas without a duration and 331 with the placeholder rank 99999),
but none of those dramas could be matched with Source B, so they are not in the joined table. The loader converts 99999 to "missing" anyway, and a median imputer stays in our code as a safe pattern.

The feature scales are very different: `genre_score` moves between about 7.5 and 8.2, while `popularity_rank` goes from 1 to more than 10,000.
"""),
md("""
## Step 3 - Exploratory figures (three figures)
"""),
code("""
fig, axes = plt.subplots(1, 5, figsize=(16, 3))
for ax, col in zip(axes, RAW_FEATURES):
    ax.hist(dramas[col], bins=30, color="#c97b8e"); ax.set_title(col)
fig.suptitle("Figure 1 - feature distributions in original units")
plt.tight_layout(); plt.show()
print("Skewness:", dramas[RAW_FEATURES].skew().round(2).to_dict())
"""),
md(f"""
**Figure 1 - three observations.** (1) `runtime_min` has **two peaks**: short web dramas (about 10-25 minutes) and standard TV dramas (about 60-70 minutes).
(2) `popularity_rank` and `episodes` are strongly right-skewed (skewness {U['dataset']['skew_before_log']['popularity_rank']:.1f} and {U['dataset']['skew_before_log']['episodes']:.1f}): a few dramas have 100+ episodes.
(3) `public_rating` and `genre_score` are roughly bell-shaped and narrow.

**Feature engineering.** Distance-based methods are dominated by extreme values, so we use `log10` for the two skewed columns:
`popularity_log` and `episodes_log` (skewness after the transformation: {U['dataset']['skew_after_log']['popularity_log']:.1f} and {U['dataset']['skew_after_log']['episodes_log']:.1f}).
"""),
code("""
dramas["popularity_log"] = np.log10(dramas["popularity_rank"])
dramas["episodes_log"] = np.log10(dramas["episodes"])
FEATURES = ["genre_score", "runtime_min", "popularity_log", "episodes_log", "public_rating"]
X = dramas[FEATURES].copy()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sc = axes[0].scatter(dramas["runtime_min"], dramas["popularity_rank"], c=dramas["public_rating"], cmap="RdPu", s=10)
axes[0].set_yscale("log"); axes[0].set_xlabel("runtime_min"); axes[0].set_ylabel("popularity_rank (log scale)")
axes[0].set_title("Figure 2 - runtime vs popularity, colour = public rating")
fig.colorbar(sc, ax=axes[0])
corr = X.corr()
im = axes[1].imshow(corr, cmap="RdPu", vmin=-1, vmax=1)
axes[1].set_xticks(range(5), FEATURES, rotation=30, ha="right"); axes[1].set_yticks(range(5), FEATURES)
for i in range(5):
    for j in range(5):
        axes[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
axes[1].grid(False); axes[1].set_title("Figure 3 - correlation matrix"); fig.colorbar(im, ax=axes[1])
plt.tight_layout(); plt.show()
"""),
md("""
**Figure 2.** Two vertical bands are visible (short and long episodes). Dark points (high rating) sit at the bottom: highly rated dramas are also the popular ones.
**Figure 3.** `public_rating` and `popularity_log` are clearly related (negative sign because rank 1 is the best). `episodes_log` is almost independent of the other features.
Correlated features mean that PCA should be able to compress the data.
"""),
md("""
## Step 4 - Standardize the features

K-Means, DBSCAN, PCA, t-SNE and UMAP all work with distances or variances. Without scaling, the feature with the largest numbers
(`runtime_min`, standard deviation about 24) would decide everything and `genre_score` (standard deviation about 0.1) would be ignored.
`StandardScaler` gives every feature mean 0 and standard deviation 1. This is exploratory analysis of one fixed table, so the scaler is fitted on all rows (there is no test set and no target).
"""),
code("""
X_imputed = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=FEATURES)   # safe pattern; nothing is missing
scaler = StandardScaler()
Xs = scaler.fit_transform(X_imputed)
print("Means after scaling:", np.round(Xs.mean(axis=0), 3))
print("SDs after scaling:  ", np.round(Xs.std(axis=0), 3))
"""),
md("""
### Step 4.1 - Data quality and scaling experiments

**Scaling experiment.** Same K and same seed, raw units versus standardized units.

**Imputation experiment.** Our table has no gaps, so to compare strategies we hide 5% of the `runtime_min` values **on purpose**
(this is an artificial test, clearly separated from the real analysis) and fill them with the mean or with the median.
"""),
code("""
K_TEST = 4
raw_labels = KMeans(n_clusters=K_TEST, n_init=10, random_state=SEED).fit_predict(X_imputed)
scaled_labels = KMeans(n_clusters=K_TEST, n_init=10, random_state=SEED).fit_predict(Xs)
print("Raw silhouette:   ", round(silhouette_score(X_imputed, raw_labels), 3))
print("Scaled silhouette:", round(silhouette_score(Xs, scaled_labels), 3))
print("Agreement between the two clusterings (ARI):", round(adjusted_rand_score(raw_labels, scaled_labels), 3))
print("Raw standard deviations:", X_imputed.std().round(3).to_dict())

rng = np.random.default_rng(SEED)
hidden = rng.choice(len(X), size=int(0.05 * len(X)), replace=False)
X_holes = X_imputed.copy()
X_holes.loc[hidden, "runtime_min"] = np.nan
for strategy in ["mean", "median"]:
    filled = SimpleImputer(strategy=strategy).fit_transform(X_holes)
    filled_scaled = StandardScaler().fit_transform(filled)
    labels = KMeans(n_clusters=K_TEST, n_init=10, random_state=SEED).fit_predict(filled_scaled)
    print(f"{strategy:6s} imputation of {len(hidden)} hidden values -> silhouette",
          round(silhouette_score(filled_scaled, labels), 3),
          "| ARI vs complete data", round(adjusted_rand_score(scaled_labels, labels), 3))
"""),
md(f"""
**What the result means.**

* Raw data gives a *higher* silhouette ({sc['raw_silhouette']:.3f}) than scaled data ({sc['scaled_silhouette']:.3f}). This does **not** mean raw is better:
  in raw units the clustering looks almost only at `runtime_min`, which has two clear peaks, so the groups are easy to separate but ignore the other features.
  The two clusterings agree only moderately (ARI = {sc['ari_raw_vs_scaled']:.2f}). We keep the standardized version because we want all five features to matter.
* Imputation: with 5% hidden values, mean imputation keeps the clusters almost unchanged (ARI = {imp['mean']['ari_vs_complete_data']:.2f}) and median imputation changes them a little more
  (ARI = {imp['median']['ari_vs_complete_data']:.2f}). Reason: `runtime_min` has two peaks, and the median ({imp['median']['fill_value']:.0f} min) puts every filled drama into the "long" group,
  while the mean ({imp['mean']['fill_value']:.0f} min) puts them between the groups. Neither is "true"; with two peaks a single fill value is always a compromise.
"""),
md("""
## Step 5 and 6 - K-Means: choose K with the elbow and silhouette methods

K-Means repeats two steps: (1) give every drama to the nearest centre (centroid); (2) move each centroid to the mean of its dramas.
We must choose K ourselves. **Inertia** (within-cluster squared distance) always goes down when K grows, so we look for an "elbow".
The **silhouette score** (from -1 to 1) compares how close a point is to its own cluster and to the nearest other cluster; higher is better.
"""),
code("""
ks = list(range(2, 9))
inertias, silhouettes, smallest = [], [], []
for k in ks:
    model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    labels = model.fit_predict(Xs)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(Xs, labels))
    smallest.append(np.bincount(labels).min())

display(pd.DataFrame({"K": ks, "inertia": inertias, "silhouette": silhouettes, "smallest cluster": smallest}).round(3))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
axes[0].plot(ks, inertias, marker="o", color="#7a1f3d"); axes[0].set_title("Figure 4a - elbow: inertia"); axes[0].set_xlabel("K")
axes[1].plot(ks, silhouettes, marker="o", color="#9d8cc9"); axes[1].set_title("Figure 4b - silhouette score"); axes[1].set_xlabel("K")
plt.tight_layout(); plt.show()
"""),
md(f"""
**Choosing K - honest reading of the evidence.**

* The elbow curve is smooth: there is no sharp corner. The biggest drops are from K=2 to K=3 and from K=3 to K=4; after K=4 the gains become smaller.
* The silhouette is highest at **K=2 ({kc[2]['silhouette']:.3f})**: the strongest structure in the data is simply "short web dramas" versus "full-length dramas".
  For K=3 it is {kc[3]['silhouette']:.3f}, for **K=4 it is {kc[4]['silhouette']:.3f}** (a small second peak), and then it falls ({kc[5]['silhouette']:.3f} at K=5).
* **We choose K = 4.** K=2 is statistically the cleanest answer, but it only repeats what the runtime histogram already showed. K=4 costs very little silhouette
  ({kc[2]['silhouette']:.3f} -> {kc[4]['silhouette']:.3f}), sits where the elbow curve flattens, and gives four groups that are easy to describe.
  All silhouette values are low (about 0.3), so the groups **overlap**: they are useful segments, not sharply separated natural classes.
"""),
code("""
K = 4
kmeans = KMeans(n_clusters=K, n_init=10, random_state=SEED)
km_labels = kmeans.fit_predict(Xs)
print("Cluster sizes:", pd.Series(km_labels).value_counts().sort_index().to_dict())
print("Inertia:", round(kmeans.inertia_, 1), "| silhouette:", round(silhouette_score(Xs, km_labels), 3))
"""),
md("""
### Task 6.1 - K-Means stability and cluster profiles

A centroid is just the arithmetic mean of the points in a cluster (first line below). K-Means starts from random centres, so we repeat it with five seeds.
Cluster numbers are arbitrary names, so we compare runs with the Adjusted Rand Index (ARI: 1 = identical grouping), not by the numbers.
Profiles are shown in the **original units**.
"""),
code("""
points = X_imputed.iloc[:3, :2].to_numpy()
print("Manual centroid of the first three dramas (genre_score, runtime_min):", points.mean(axis=0).round(3))

rows = []
for seed in range(5):
    m = KMeans(n_clusters=K, n_init=10, random_state=seed)
    lab = m.fit_predict(Xs)
    rows.append({"seed": seed, "inertia": m.inertia_, "silhouette": silhouette_score(Xs, lab),
                 "ARI vs seed 42": adjusted_rand_score(km_labels, lab)})
display(pd.DataFrame(rows).round(3))

profile_cols = ["genre_score", "runtime_min", "popularity_rank", "episodes", "public_rating", "year"]
profiles = dramas.assign(cluster=km_labels).groupby("cluster")[profile_cols].mean().round(2)
profiles.insert(0, "size", pd.Series(km_labels).value_counts().sort_index())
profiles["median popularity_rank"] = dramas.assign(cluster=km_labels).groupby("cluster")["popularity_rank"].median()
display(profiles)

z_means = pd.DataFrame(Xs, columns=FEATURES).assign(cluster=km_labels).groupby("cluster").mean()
fig, ax = plt.subplots(figsize=(7.5, 3.2))
im = ax.imshow(z_means.values, cmap="RdPu", aspect="auto")
ax.set_xticks(range(5), FEATURES, rotation=20); ax.set_yticks(range(K), [f"cluster {c}" for c in range(K)])
for i in range(K):
    for j in range(5):
        ax.text(j, i, f"{z_means.values[i, j]:+.2f}", ha="center", va="center")
ax.grid(False); ax.set_title("Figure 5 - cluster profiles (mean z-score)"); fig.colorbar(im)
plt.tight_layout(); plt.show()

for c in range(K):
    members = dramas[km_labels == c]
    top_genres = (members["genres"].str.split("|").explode().value_counts() / len(members)).head(4).round(2).to_dict()
    print(f"cluster {c}: top genres {top_genres}")
    print("           most popular examples:", members.sort_values("popularity_rank")["title"].head(5).tolist())
"""),
md(f"""
**What the result means.**

* **Stability:** five different seeds give almost the same inertia and silhouette, and the groupings agree strongly
  (ARI between {min(r['ari_vs_seed_42'] for r in km['seed_stability']):.2f} and 1.00). The solution does not depend much on the random start.
* **Cluster profiles** (means in original units; the description is generated from the z-scores, not invented):

| Cluster | Size | Runtime (min) | Episodes | Median popularity rank | Public rating | Genre score | Data-derived description |
|---|---|---|---|---|---|---|---|
{profile_rows}

* In plain words: one group is **short web dramas**, one is **long-running serials** with many episodes, one is the **popular, highly rated hits**, and one is the
  **standard-format dramas with average popularity**. These names are our reading of the numbers; the algorithm only returns the numbers 0-3.
* Genres were **not** used for clustering (only the numeric genre score), so the top-genre lists are an external description. Romance, Drama and Comedy lead in every cluster, which shows that these groups are about *format and popularity*, not about genre.
"""),
md("""
## Step 7 - DBSCAN density clustering

DBSCAN does not need K. A point is a **core point** if at least `min_samples` points lie within distance `eps`. Connected core points form a cluster;
points that belong to no dense region get the label **-1 (noise)**. To choose `eps` we use the k-distance plot: the distance from each drama to its 10th nearest neighbour.
A common rule of thumb is `min_samples = 2 x number of features = 10`.
"""),
code("""
MIN_SAMPLES = 10
k_distance = np.sort(NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(Xs).kneighbors(Xs)[0][:, -1])
print("k-distance percentiles (50, 75, 90, 95, 99):", np.percentile(k_distance, [50, 75, 90, 95, 99]).round(2))

EPS = 0.8
db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
db_labels = db.fit_predict(Xs)
print(pd.Series(db_labels).value_counts().sort_index().to_string())
print("Clusters:", len(set(db_labels) - {-1}), "| noise points:", int((db_labels == -1).sum()))

plt.figure(figsize=(6, 3.3))
plt.plot(k_distance, color="#7a1f3d"); plt.axhline(EPS, color="gray", linestyle="--", label=f"eps = {EPS}")
plt.xlabel("dramas sorted by distance"); plt.ylabel("distance to 10th neighbour")
plt.title("Figure 6 - k-distance plot"); plt.legend(); plt.show()
"""),
md("""
### Step 7.1 - DBSCAN sensitivity (15 settings)

We try every combination of five `eps` values and three `min_samples` values. Silhouette is computed **without the noise points**
and only when there are at least two clusters (otherwise it is not defined).
"""),
code("""
rows = []
for eps in [0.5, 0.6, 0.7, 0.8, 1.0]:
    for ms in [5, 10, 15]:
        lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xs)
        groups = len(set(lab) - {-1})
        noise = int((lab == -1).sum())
        keep = lab != -1
        sil = silhouette_score(Xs[keep], lab[keep]) if groups >= 2 else np.nan
        rows.append((eps, ms, groups, noise, round(noise / len(lab), 3), sil))
grid = pd.DataFrame(rows, columns=["eps", "min_samples", "clusters", "noise", "noise share", "silhouette (no noise)"])
display(grid.round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ms, color in zip([5, 10, 15], PALETTE):
    part = grid[grid["min_samples"] == ms]
    axes[0].plot(part["eps"], part["clusters"], marker="o", color=color, label=f"min_samples={ms}")
    axes[1].plot(part["eps"], part["noise"], marker="o", color=color, label=f"min_samples={ms}")
axes[0].set_title("Figure 7a - number of clusters"); axes[1].set_title("Figure 7b - number of noise points")
for ax in axes:
    ax.set_xlabel("eps"); ax.legend()
plt.tight_layout(); plt.show()

print("K-Means cluster (rows) vs DBSCAN cluster (columns, -1 = noise):")
display(pd.crosstab(pd.Series(km_labels, name="kmeans"), pd.Series(db_labels, name="dbscan")))
keep = db_labels != -1
print("ARI between K-Means and DBSCAN on non-noise dramas:", round(adjusted_rand_score(km_labels[keep], db_labels[keep]), 3))
display(dramas.assign(dbscan=db_labels).groupby("dbscan")[["runtime_min", "episodes", "popularity_rank", "public_rating", "genre_score"]].mean().round(2))
"""),
md(f"""
**What the result means.**

* **Main setting** `eps = {db['eps']}`, `min_samples = {db['min_samples']}`: **{db['clusters']} clusters and {db['noise']} noise points** ({db['noise'] / U['dataset']['observations']:.0%} of the dramas).
  The k-distance curve bends between the 75th percentile ({db['k_distance_percentiles']['75']:.2f}) and the 95th percentile ({db['k_distance_percentiles']['95']:.2f}), so 0.8 is a reasonable radius.
* **Sensitivity is high.** With `eps = 0.5, min_samples = 5` DBSCAN finds {grid[(0.5, 5)]['clusters']} small clusters and {grid[(0.5, 5)]['noise']} noise points;
  with `eps = 1.0, min_samples = 10` it finds {grid[(1.0, 10)]['clusters']} cluster and only {grid[(1.0, 10)]['noise']} noise points. Small eps = many tiny groups and a lot of noise; large eps = everything merges.
  There is no single "correct" setting - the density of this dataset changes smoothly.
* **DBSCAN vs K-Means.** DBSCAN separates the short web dramas and the long serials clearly (they are dense and far from the rest), but it puts the K-Means "hits" and "standard" groups into **one** big cluster,
  because there is no empty space between them. On the dramas that DBSCAN does not call noise, the two methods agree moderately (ARI = {db['ari_with_kmeans_excluding_noise']:.2f}).
  K-Means must give every drama a cluster; DBSCAN is allowed to say "this one does not belong anywhere".
* **Noise is not an error.** A noise point is only a drama in a sparse region for this eps and min_samples.
"""),
md("""
## Step 8 - PCA and explained variance

PCA finds new axes (principal components) that are straight-line mixtures of the standardized features.
PC1 keeps as much variance as possible, PC2 keeps as much as possible of the rest, and so on. It is linear and deterministic.
"""),
code("""
pca_full = PCA().fit(Xs)
ratio = pca_full.explained_variance_ratio_
print("Explained variance ratio:", ratio.round(3))
print("Cumulative variance:     ", np.cumsum(ratio).round(3))
loadings = pd.DataFrame(pca_full.components_[:2], columns=FEATURES, index=["PC1", "PC2"])
display(loadings.round(2))

pca = PCA(n_components=2)
Xp = pca.fit_transform(Xs)
print("Total retained by 2 components:", round(pca.explained_variance_ratio_.sum(), 3))

def scatter(ax, xy, labels, title, noise_label=None):
    for c in sorted(set(labels)):
        mask = labels == c
        if c == noise_label:
            ax.scatter(xy[mask, 0], xy[mask, 1], s=9, c="#3a3942", marker="x", label="noise", alpha=0.7)
        else:
            ax.scatter(xy[mask, 0], xy[mask, 1], s=9, c=PALETTE[c % len(PALETTE)], label=f"cluster {c}", alpha=0.8)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].bar(range(1, 6), ratio, color="#c97b8e", label="per component")
axes[0].plot(range(1, 6), np.cumsum(ratio), marker="o", color="#7a1f3d", label="cumulative")
axes[0].set_xlabel("principal component"); axes[0].set_title("Figure 8a - explained variance"); axes[0].legend()
scatter(axes[1], Xp, km_labels, "Figure 8b - PCA projection coloured by K-Means"); axes[1].legend(markerscale=2)
plt.tight_layout(); plt.show()
"""),
md(f"""
**What the result means.**

* PC1 keeps **{pca['explained_variance_ratio'][0]:.1%}** and PC2 keeps **{pca['explained_variance_ratio'][1]:.1%}** of the standardized variance: **{pca['retained_2d']:.1%} in two dimensions**.
  About {1 - pca['retained_2d']:.0%} of the information is not visible in the 2-D picture; three components would keep {pca['cumulative'][2]:.1%}.
* **Loadings.** PC1 mixes `public_rating`, `runtime_min`, `genre_score` (positive) and `popularity_log` (negative): it is a "mainstream success" axis - long, well-rated, popular dramas on one side, short and less-known ones on the other.
  PC2 is almost only `episodes_log` ({pca['loadings']['PC2']['episodes_log']:.2f}): the "number of episodes" axis. Loadings describe directions of variance, not causes.
* In the scatter plot the K-Means groups are visible as regions but they **touch and overlap**. Overlap in 2-D does not prove overlap in all five dimensions.
"""),
md("""
## Step 9 - t-SNE visualization

t-SNE is a non-linear method. It tries to keep **neighbours** together: dramas that are close in five dimensions should be close on the map.
It does not keep global distances: the size of a blob and the space between blobs have no direct meaning.
Colours are the K-Means labels - they are added for reading only; t-SNE itself never sees them.
"""),
code("""
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, init="pca", learning_rate="auto")
Xt = tsne.fit_transform(Xs)
fig, ax = plt.subplots(figsize=(6, 4.5))
scatter(ax, Xt, km_labels, "Figure 9 - t-SNE (perplexity = 30), coloured by K-Means"); ax.legend(markerscale=2)
plt.show()
"""),
md("""
**What the result means.** t-SNE draws several islands. The short web dramas and the long serials form their own islands, which agrees with K-Means and DBSCAN.
But the islands are a *visualization*: t-SNE tends to cut continuous data into separate blobs, so a gap on this map is **not proof** of a real cluster.
We trust a group only when several methods agree.
"""),
md("""
## Step 10 - UMAP visualization

UMAP is another non-linear method. It builds a graph of nearest neighbours and then draws that graph in 2-D.
`n_neighbors` balances local detail and the bigger picture; `min_dist` controls how tightly points are packed.
"""),
code("""
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED)
Xu = reducer.fit_transform(Xs)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
scatter(axes[0], Xp, km_labels, "PCA (linear)")
scatter(axes[1], Xt, km_labels, "t-SNE (perplexity = 30)")
scatter(axes[2], Xu, km_labels, "UMAP (n_neighbors = 15)")
axes[0].legend(markerscale=2)
fig.suptitle("Figure 10 - three 2-D views of the same data, coloured by K-Means cluster")
plt.tight_layout(); plt.show()
"""),
md(f"""
**Comparing PCA, t-SNE and UMAP.**

| | PCA | t-SNE | UMAP |
|---|---|---|---|
| Type | linear | non-linear | non-linear |
| Keeps | global variance directions | local neighbourhoods | local neighbourhoods, some global layout |
| Axes have a meaning? | yes (loadings) | no | no |
| Same result every run? | yes | only with a fixed seed | only with a fixed seed |
| In our data | one continuous cloud, {pca['retained_2d']:.0%} of variance | several separate islands | compact, well separated islands |

All three agree on the main story: short web dramas and long serials stand apart, while "hits" and "standard" dramas are neighbours that blend into each other.
**PCA is the most interpretable** (its axes can be explained and its distances are honest), but it shows less separation.
t-SNE and UMAP look more "clustered", but that look is partly created by the method: distances between islands and island sizes must not be interpreted.
"""),
md("""
### Step 10.1 - Projection sensitivity

We change one setting of each method. If the picture changes a lot, we should not over-interpret any single picture.
"""),
code("""
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, perplexity in zip(axes[0], [10, 30, 50]):
    z = Xt if perplexity == 30 else TSNE(n_components=2, perplexity=perplexity, random_state=SEED,
                                         init="pca", learning_rate="auto").fit_transform(Xs)
    scatter(ax, z, km_labels, f"t-SNE perplexity = {perplexity}")
for ax, n_neighbors in zip(axes[1], [5, 15, 50]):
    z = Xu if n_neighbors == 15 else umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=SEED).fit_transform(Xs)
    scatter(ax, z, km_labels, f"UMAP n_neighbors = {n_neighbors}")
fig.suptitle("Figure 11 - projection sensitivity")
plt.tight_layout(); plt.show()
"""),
md("""
**What the result means.** With a small perplexity or few neighbours the maps break into many small pieces; with large values the pieces merge into bigger shapes.
The *number* of islands therefore depends on the setting. What stays stable is which dramas are neighbours (the colours stay together). Absolute positions and distances cannot be compared between plots.
"""),
md("""
## Step 11 - Isolation Forest anomaly detection

Isolation Forest builds many random trees. Each tree cuts the data with random splits. An unusual drama is separated from the others after only a few cuts,
so it gets a low score. `contamination` is **our** decision about what share of dramas to flag - it is not a measured anomaly rate.
"""),
code("""
iso = IsolationForest(contamination=0.05, random_state=SEED)
outlier_labels = iso.fit_predict(Xs)                 # +1 = normal, -1 = flagged
anomaly_score = iso.decision_function(Xs)            # lower = more unusual
print(pd.Series(outlier_labels).value_counts().to_string())

top = dramas.assign(anomaly_score=anomaly_score, dbscan_noise=(db_labels == -1)).sort_values("anomaly_score").head(10)
display(top[["title", "year", "genres", "runtime_min", "episodes", "popularity_rank", "public_rating", "anomaly_score", "dbscan_noise"]].round(3))
"""),
md("""
### Step 11.1 - Contamination values and comparison with DBSCAN noise
"""),
code("""
noise_mask = db_labels == -1
flags, rows = {}, []
for c in [0.02, 0.05, 0.10]:
    m = IsolationForest(contamination=c, random_state=SEED).fit(Xs)
    flags[c] = m.predict(Xs) == -1
    rows.append({"contamination": c, "flagged": int(flags[c].sum()),
                 "also DBSCAN noise (overlap)": int((noise_mask & flags[c]).sum()),
                 "only Isolation Forest": int((flags[c] & ~noise_mask).sum()),
                 "only DBSCAN noise": int((~flags[c] & noise_mask).sum()),
                 "disagreement": int((noise_mask != flags[c]).sum())})
display(pd.DataFrame(rows))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
scatter(axes[0], Xp, km_labels, "K-Means (K = 4)")
scatter(axes[1], Xp, db_labels, "DBSCAN (eps = 0.8, min_samples = 10)", noise_label=-1)
axes[2].scatter(Xp[~flags[0.05], 0], Xp[~flags[0.05], 1], s=9, c="#c9c7d1", label="normal")
axes[2].scatter(Xp[flags[0.05], 0], Xp[flags[0.05], 1], s=18, c="#7a1f3d", label="flagged")
axes[2].set_title("Isolation Forest (contamination = 0.05)"); axes[2].set_xticks([]); axes[2].set_yticks([]); axes[2].grid(False)
for ax in axes:
    ax.legend(markerscale=2, fontsize=8)
fig.suptitle("Figure 12 - three methods in the same PCA view")
plt.tight_layout(); plt.show()
"""),
md(f"""
**What the result means.**

| contamination | flagged | also DBSCAN noise | only Isolation Forest | only DBSCAN noise |
|---|---|---|---|---|
{iso_rows}

* With contamination = {iso_main['contamination']} the forest flags **{iso_main['flagged']} dramas**; **{iso_main['also_dbscan_noise']} of them ({iso_main['share_of_flags_that_are_noise']:.0%}) are also DBSCAN noise**.
  So the dramas that Isolation Forest finds most unusual are almost always in sparse regions too. The opposite is not true: DBSCAN marks {db['noise']} dramas as noise, far more than the forest flags, because its "noise" simply means "not in a dense area".
  The two methods measure different kinds of unusual.
* **Who is flagged?** The most unusual dramas include {top_titles}. They are of two kinds: (a) **mega-hits** - extremely popular, very highly rated, long episodes;
  (b) **extreme formats** - daily serials with 100+ episodes or tiny web dramas of a few minutes. "Anomaly" here means *rare combination of values*, not *bad drama* and not *data error*.
* **Two checks before calling something a real anomaly:** (1) open the drama's page on MyDramaList and confirm that the runtime and number of episodes are correct (not a data-entry mistake);
  (2) check whether the flag stays when the seed or the contamination value changes.
"""),
md("""
## Step 12 - Combine the methods in one summary table
"""),
code("""
summary = pd.DataFrame({
    "title": dramas["title"], "kmeans_cluster": km_labels, "dbscan_cluster": db_labels,
    "isolation_forest_flag": outlier_labels, "PC1": Xp[:, 0].round(3), "PC2": Xp[:, 1].round(3),
})
display(summary.head(10))
print(summary.groupby("kmeans_cluster").size().to_string())
display(pd.crosstab(summary["kmeans_cluster"], summary["isolation_forest_flag"]).rename(columns={-1: "flagged", 1: "normal"}))
"""),
md(f"""
## Conclusions

We analyzed **{U['dataset']['observations']:,} Korean dramas** and **5 numerical features** (genre score, runtime, popularity, episodes, public rating) built from two public Kaggle datasets scraped from MyDramaList.
After checking data quality (no missing values in the joined table), log-transforming two skewed features and standardizing, **K-Means suggested K = {km['chosen_k']}**: the silhouette is highest at K = 2 ({kc[2]['silhouette']:.3f}),
but K = 4 ({kc[4]['silhouette']:.3f}) lies where the elbow curve flattens and gives four interpretable groups (short web dramas, long serials, popular hits, standard dramas).
**DBSCAN** with eps = {db['eps']} and min_samples = {db['min_samples']} identified **{db['clusters']} groups and {db['noise']} noise points**, and its result changed strongly with the parameters.
The **first two PCA components retained {pca['retained_2d']:.1%}** of the standardized variance. The **t-SNE and UMAP** plots showed separate islands for short web dramas and long serials, subject to projection limitations.
**Isolation Forest** flagged **{iso_main['flagged']} unusual dramas** with contamination = {iso_main['contamination']}; {iso_main['also_dbscan_noise']} of them were also DBSCAN noise.
None of the unsupervised outputs independently establishes a ground-truth class or a verified anomaly.

**Evidence-based findings**
1. The strongest structure is *format*: episode length splits K-Dramas into web dramas and full-length dramas. Every method shows it.
2. Popularity, public rating and runtime move together (PC1, {pca['explained_variance_ratio'][0]:.0%} of variance). Number of episodes is a separate, independent direction (PC2).
3. "Hits" are the upper end of a continuous scale, not a separate island: DBSCAN merges them with ordinary dramas and the silhouette is low.
4. The most unusual dramas are the global mega-hits and the extreme formats.

**Limitations**
1. Only dramas present in both sources are included ({U['dataset']['observations']:,} of 1,752), so less-known dramas are under-represented.
2. Clusters overlap (silhouette about 0.3); K = 4 is a useful choice, not a discovered truth.
3. DBSCAN, t-SNE and UMAP depend strongly on their parameters; contamination is chosen by us.
4. `genre_score` is built from `public_rating`, so these two features are related by construction.
5. Popularity rank and ratings are a snapshot from 2023; Source B has no stated licence (classroom use only).

**Improvements.** Add more features (tags, network, cast popularity); encode genres directly (multi-hot) instead of one score; try a Gaussian Mixture Model or HDBSCAN, which handle overlapping and uneven densities better; check cluster stability with bootstrap samples.

**Reproducibility.** `SEED = 42` is used for K-Means, t-SNE, UMAP and Isolation Forest; DBSCAN and PCA are deterministic. Run all cells from top to bottom. Small numerical differences in t-SNE/UMAP pictures are possible between library versions.

### Answers to the worksheet questions
* **Why must a known target be excluded from clustering features?** Unsupervised learning should find structure without the answer. If the target were a feature, the groups would simply copy the label and we would learn nothing new.
* **How did scaling change the results?** Without scaling the clustering followed almost only `runtime_min` (silhouette {sc['raw_silhouette']:.2f} but only one feature used); with scaling all five features count (silhouette {sc['scaled_silhouette']:.2f}); the two groupings agree only moderately (ARI {sc['ari_raw_vs_scaled']:.2f}).
* **What supports the selected K?** The elbow curve flattens after K = 4, the silhouette has a small second peak at K = 4, the solution is stable over seeds, and the four profiles are interpretable.
* **How sensitive is DBSCAN?** Very: from {grid[(0.5, 5)]['clusters']} clusters / {grid[(0.5, 5)]['noise']} noise points to {grid[(1.0, 10)]['clusters']} cluster / {grid[(1.0, 10)]['noise']} noise points inside our grid.
* **How much variance does PCA keep, and what do t-SNE/UMAP not preserve?** {pca['retained_2d']:.1%} in two components. t-SNE and UMAP do not preserve global distances, cluster sizes or densities.
* **Where do DBSCAN noise and Isolation Forest agree?** {iso_main['also_dbscan_noise']} of the {iso_main['flagged']} flagged dramas are also noise; {iso_main['only_dbscan_noise']} noise points are not flagged by the forest.
* **Two limitations?** Overlapping clusters with low silhouette, and strong parameter sensitivity of DBSCAN/t-SNE/UMAP (plus the selection of dramas present in both sources).

### References
* Charuchinda, C. (2023). *Korean Drama from 2015-2023 with Actors & Reviews* [Data set]. Kaggle. CC0 Public Domain. https://www.kaggle.com/datasets/chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist
* Rizki, N. (2023). *Top Korean Drama List (~1500)* [Data set]. Kaggle. https://www.kaggle.com/datasets/noorrizki/top-korean-drama-list-1500
* MyDramaList. https://mydramalist.com (original source of both datasets)
* Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
* van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9, 2579-2605.
* McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
* Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *2008 Eighth IEEE International Conference on Data Mining*, 413-422.
* Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. *KDD-96*, 226-231.
"""),
code("""
# Key numbers of this run, saved so that the report and the website can be checked against the notebook.
import json
check = {"observations": int(len(dramas)),
         "k_silhouette": {int(k): float(s) for k, s in zip(ks, silhouettes)},
         "kmeans_sizes": [int(v) for v in np.bincount(km_labels)],
         "dbscan_clusters": int(len(set(db_labels) - {-1})), "dbscan_noise": int(noise_mask.sum()),
         "pca_ratio": [float(v) for v in ratio],
         "iso_flagged": {str(c): int(f.sum()) for c, f in flags.items()},
         "iso_overlap_noise": {str(c): int((f & noise_mask).sum()) for c, f in flags.items()}}
with open("p2_notebook_results.json", "w") as f:
    json.dump(check, f, indent=2)
print(json.dumps(check, indent=2))
"""),
]

out = ROOT / "notebooks"
nbf.write(notebook(p1), out / "Project1_Supervised_KDrama.ipynb")
nbf.write(notebook(p2), out / "Project2_Unsupervised_KDrama.ipynb")
print("Notebooks written to", out)
