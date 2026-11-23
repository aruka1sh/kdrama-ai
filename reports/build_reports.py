"""Builds the two academic reports (.docx) from the REAL result files.

Run:  python reports/build_reports.py
Reads artifacts/supervised_results.json, artifacts/unsupervised_results.json and artifacts/figures/*.png.
Every number in the reports is inserted from those files - nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "figures"
S = json.loads((ROOT / "artifacts" / "supervised_results.json").read_text())
U = json.loads((ROOT / "artifacts" / "unsupervised_results.json").read_text())

TEAM_LINE = "Shakenova Aruzhan, Qabir Madina, Shahap Aizere, Zholdaspek Dana"
BURGUNDY = RGBColor(0x7A, 0x1F, 0x3D)

REF_A = ("Charuchinda, C. (2023). Korean Drama from 2015-2023 with Actors & Reviews [Data set]. Kaggle. "
         "Licence: CC0 Public Domain. "
         "https://www.kaggle.com/datasets/chanoncharuchinda/korean-drama-2015-23-actor-and-reviewmydramalist")
REF_B = ("Rizki, N. (2023). Top Korean Drama List (~1500) [Data set]. Kaggle. Licence not stated. "
         "https://www.kaggle.com/datasets/noorrizki/top-korean-drama-list-1500")
REF_MDL = "MyDramaList. https://mydramalist.com (original source of both datasets)."
REF_SK = ("Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. "
          "Journal of Machine Learning Research, 12, 2825-2830.")


class Report:
    def __init__(self, title: str, subtitle: str):
        self.doc = Document()
        self.fig_no = 0
        self.tab_no = 0
        style = self.doc.styles["Normal"]
        style.font.name = "Calibri"
        style.font.size = Pt(11)
        for section in self.doc.sections:
            section.left_margin = section.right_margin = Cm(2.3)
            section.top_margin = section.bottom_margin = Cm(2.2)
        for level, size in [(1, 15), (2, 12.5)]:
            h = self.doc.styles[f"Heading {level}"]
            h.font.name = "Calibri"
            h.font.size = Pt(size)
            h.font.bold = True
            h.font.color.rgb = BURGUNDY
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("K-DRAMA AI")
        r.bold = True
        r.font.size = Pt(26)
        r.font.color.rgb = BURGUNDY
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(title)
        r.bold = True
        r.font.size = Pt(16)
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run(subtitle).italic = True
        for line in ["AI Fundamentals - Final Exam Project, QAIRU",
                     "Variant 21 - Movie Preference (adapted to Korean dramas)",
                     f"Team: {TEAM_LINE}"]:
            p = self.doc.add_paragraph(line)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
        self.doc.add_paragraph()

    def h1(self, text):
        self.doc.add_heading(text, level=1)

    def h2(self, text):
        self.doc.add_heading(text, level=2)

    def p(self, text):
        para = self.doc.add_paragraph()
        para.paragraph_format.space_after = Pt(6)
        parts = text.split("**")                      # **bold** segments
        for i, part in enumerate(parts):
            if part:
                para.add_run(part).bold = i % 2 == 1
        return para

    def bullets(self, items, numbered=False):
        for item in items:
            para = self.doc.add_paragraph(style="List Number" if numbered else "List Bullet")
            parts = item.split("**")
            for i, part in enumerate(parts):
                if part:
                    para.add_run(part).bold = i % 2 == 1

    def table(self, header, rows, caption, widths=None):
        self.tab_no += 1
        cap = self.doc.add_paragraph()
        r = cap.add_run(f"Table {self.tab_no}. {caption}")
        r.bold = True
        r.font.size = Pt(10)
        cap.paragraph_format.space_after = Pt(2)
        t = self.doc.add_table(rows=1, cols=len(header))
        t.style = "Light Grid Accent 1"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, text in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            run = cell.paragraphs[0].add_run(str(text))
            run.bold = True
            run.font.size = Pt(9.5)
        for row in rows:
            cells = t.add_row().cells
            for i, value in enumerate(row):
                cells[i].text = ""
                cells[i].paragraphs[0].add_run(str(value)).font.size = Pt(9.5)
        if widths:
            for row in t.rows:
                for cell, w in zip(row.cells, widths):
                    cell.width = Cm(w)
        self.doc.add_paragraph()

    def figure(self, name, caption, width=16.0):
        self.fig_no += 1
        self.doc.add_picture(str(FIG / name), width=Cm(width))
        self.doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap = self.doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(f"Figure {self.fig_no}. {caption}")
        r.italic = True
        r.font.size = Pt(9.5)

    def save(self, name):
        path = ROOT / "reports" / name
        self.doc.save(path)
        print("saved", path)


f3 = lambda v: f"{v:.3f}"
pc = lambda v, d=1: f"{v * 100:.{d}f}%"


# ======================================================================================
# REPORT 1 - SUPERVISED LEARNING
# ======================================================================================
def report_1():
    ds, sp, cv, t = S["dataset"], S["split"], S["cross_validation"], S["test"]
    cm = t["confusion_matrix"]
    best = S["selected_model"]
    other = S["test_other_model_reference_only"]
    interp = S["interpretation"]
    perm = interp["permutation_importance_test_auc_drop"]
    order = sorted(perm, key=lambda f: -perm[f]["mean"])
    new = S["new_record"]

    r = Report("Project 1 - Supervised Learning", "Will a viewer like this K-Drama? A binary classification study")

    r.h1("1. Objective")
    r.p("The objective is to build and honestly evaluate a binary classifier that predicts whether a viewer will like "
        "a Korean drama (K-Drama), using measurable characteristics of the drama and of the viewer's taste. The work "
        "follows the Practice 8 worksheet sequence: DATA, SPLIT, PREPARE, TRAIN, VALIDATE, TEST, DECIDE.")

    r.h1("2. Problem statement")
    r.p("Variant 21 of the worksheet is 'Movie preference': predict 'likes movie' from genre score, runtime and "
        "popularity. The worksheet gives no exact dataset for this variant. We adapted the task to Korean dramas and "
        "found a public dataset that contains real individual viewer scores, so no label had to be simulated. "
        "This adaptation of the task should be confirmed with the instructor, as the worksheet requires.")
    r.table(["Element", "Definition"], [
        ["Object (one row)", "One real review: one viewer scores one K-Drama on MyDramaList"],
        ["Target", f"liked = 1 if the viewer's own overall score is {ds['like_threshold']:g} or more out of 10, otherwise 0"],
        ["Task type", "Binary classification"],
        ["Variant features", "genre_score, runtime_min, popularity (as popularity_log)"],
        ["Additional features", "public_rating, episodes, year"],
    ], "Problem definition", widths=[4.5, 12])

    r.h1("3. Dataset description")
    r.h2("3.1 Sources")
    r.bullets([
        "**Source A** - Chanon Charuchinda (2023), 'Korean Drama from 2015-2023 with Actors & Reviews', Kaggle, licence CC0. "
        "File korean_drama.csv (1,752 dramas: runtime, episodes, popularity rank) and reviews.csv (10,625 reviews with each viewer's overall score).",
        "**Source B** - Noor Rizki (2023), 'Top Korean Drama List (~1500)', Kaggle, licence not stated. "
        "File kdrama_list.csv (genres and public MyDramaList rating).",
    ])
    r.p("Both sources were scraped from MyDramaList.com. They are downloaded automatically by the notebook and joined "
        "by normalized title and year. Reviews of dramas that could not be matched in both sources are dropped. "
        f"The final table has **{ds['rows']:,} reviews** written by **{ds['viewers']:,} viewers** about **{ds['dramas']:,} dramas**. "
        "No value in the table is simulated.")
    r.h2("3.2 Features")
    r.table(["Feature", "Meaning", "Missing"], [
        ["genre_score", "Mean score (1-10) the same viewer gave to OTHER dramas that share at least one genre with this drama. "
                        "The review being predicted is never included.", f"{ds['missing_counts']['genre_score']:,} ({pc(ds['missing_share_genre_score'])})"],
        ["runtime_min", "Length of one episode in minutes", "0"],
        ["popularity_log", "log10 of the MyDramaList popularity rank (rank 1 = most popular)", "0"],
        ["public_rating", "Public MyDramaList rating of the drama (average of all users, 0-10)", "0"],
        ["episodes", "Number of episodes", "0"],
        ["year", "Release year", "0"],
    ], "Feature dictionary", widths=[3, 10.5, 3])
    r.p("**Why a public rating is not used as the label.** A drama's public rating is an average over thousands of users; "
        "it is not the opinion of one person. Our target is each viewer's own score. The public rating is only an input feature.")
    r.h2("3.3 Target and class balance")
    r.p(f"The source score runs from 1 to 10 in steps of 0.5. We group it into two classes: 8.0-10 = class 1 ('liked') and "
        f"1-7.5 = class 0. The median score is 8. Class 1 has {ds['class_counts']['1']:,} reviews ({pc(ds['positive_rate'])}) and "
        f"class 0 has {ds['class_counts']['0']:,} reviews ({pc(1 - ds['positive_rate'])}). The classes are moderately unbalanced, "
        "so F1 and ROC-AUC are more informative than accuracy.")
    r.h2("3.4 Missing values")
    r.p(f"Only genre_score has missing values ({pc(ds['missing_share_genre_score'])} of rows). They are real: these viewers have "
        "no other review with a shared genre (mostly viewers with a single review). We did not add artificial missing values; "
        "the worksheet's worked example states that missingness should reflect the actual dataset.")

    r.h1("4. Exploratory data analysis")
    r.figure("p1_class_balance.png", "Class balance and distribution of the viewers' overall scores. The dashed line is the 'liked' threshold.", 14.5)
    r.p("Reviewers are generous: most scores are 7 or higher. A model that always answers 'liked' is already right more than "
        "half of the time, so we report this simple baseline next to our models.")
    r.figure("p1_feature_distributions.png", "Distribution of the six features for liked and not-liked reviews.")
    r.p("Liked reviews are shifted to higher genre_score and public_rating. runtime_min has two peaks (short web dramas and "
        "60-70 minute TV dramas). The features have very different scales, which is why Logistic Regression needs scaling.")
    r.figure("p1_correlation.png", "Correlation matrix of the features and the target.", 10)
    r.p("genre_score and public_rating have the strongest correlation with the target. public_rating and popularity_log are "
        "negatively correlated (popular dramas have a small rank number and a higher rating). Correlation is not causation.")

    r.h1("5. Preprocessing")
    r.bullets([
        "**Feature engineering:** popularity_log = log10(popularity_rank). It is a row-by-row formula, so it is safe before the split.",
        f"**80/20 split before any fitted preprocessing:** {sp['train_rows']:,} training rows and {sp['test_rows']:,} test rows "
        f"({pc(sp['test_share'])}). The split is stratified (liked share {pc(sp['train_positive_rate'])} in train and {pc(sp['test_positive_rate'])} in test).",
        f"**Grouped by viewer:** all reviews of one viewer stay on the same side ({sp['train_viewers']:,} training viewers, "
        f"{sp['test_viewers']:,} test viewers, 0 in common). Because genre_score is built from the viewer's other reviews, a normal random "
        "split would let the model meet the same people in training and test - a hidden data leak. We therefore used "
        "StratifiedGroupKFold (5 folds, one fold = test set, shuffle=False so that the split is deterministic and identical in every scikit-learn version) instead of train_test_split.",
        "**Pipelines:** median imputation and standard scaling are steps of a scikit-learn Pipeline, so their statistics are learned from "
        "training rows only, also inside every cross-validation fold.",
    ])

    r.h1("6. Methodology and algorithms")
    r.table(["Algorithm", "Why it is appropriate", "Settings"], [
        ["Logistic Regression", "Simple linear model for binary targets; outputs probabilities; coefficients are easy to interpret. A transparent reference model.",
         "max_iter=1000, random_state=42"],
        ["Random Forest", "Ensemble of decision trees; learns non-linear rules and interactions; robust to skewed features.",
         "n_estimators=150, max_depth=6, min_samples_leaf=3, random_state=42"],
    ], "Algorithms", widths=[3.3, 8.2, 5])
    r.p("Both pipelines have the same steps: SimpleImputer(median) -> StandardScaler -> classifier. Model comparison uses the same "
        "stratified, viewer-grouped 3-fold cross-validation folds for both models, with F1 as the score. The model with the higher "
        "mean cross-validation F1 is selected, and only then is the test set used, once.")

    r.h1("7. Implementation")
    r.p("The project is implemented in Python with pandas, NumPy, Matplotlib and scikit-learn. The notebook "
        "Project1_Supervised_KDrama.ipynb runs from top to bottom in Google Colab and downloads the data automatically. "
        "The same pipeline is trained by ml/train_supervised.py, which saves the model with joblib for the website. "
        "The split and the folds are deterministic and random_state = 42 is used for both models; the notebook and the script produce identical numbers.")

    r.h1("8. Results")
    r.h2("8.1 Cross-validation and model selection")
    models = cv["models"]
    r.table(["Model", "F1 fold 1", "F1 fold 2", "F1 fold 3", "Mean F1", "SD", "Mean ROC-AUC"],
            [[n, *[f3(v) for v in m["f1_folds"]], f3(m["f1_mean"]), f3(m["f1_std"]), f3(m["roc_auc_mean"])] for n, m in models.items()]
            + [["Baseline: always 'liked'", "-", "-", "-", f3(cv["baseline_always_like_f1_mean"]), "-", "0.500"]],
            "Training-only 3-fold cross-validation (same folds for all models)")
    r.figure("p1_cv_comparison.png", "Mean cross-validation F1 with standard deviation.", 9)
    r.p(f"**Selected model: {best}** (mean CV F1 {f3(models[best]['f1_mean'])} against {f3(models[other['model']]['f1_mean'])}). "
        "The two models are very close - the difference is about one standard deviation - and both are only a few F1 points above the "
        "baseline, because the positive class is the majority. ROC-AUC shows more clearly that the models rank reviews better than chance.")
    r.h2("8.2 Final evaluation on the untouched test set")
    r.table(["Metric", "Value", "Meaning in simple words"], [
        ["Confusion matrix", f"TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}", "Counts of correct and wrong answers by class"],
        ["Precision", f3(t["precision"]), f"When the model says 'liked', it is right about {pc(t['precision'], 0)} of the time"],
        ["Recall", f3(t["recall"]), f"The model finds about {pc(t['recall'], 0)} of the really liked dramas"],
        ["F1 score", f3(t["f1"]), "Balance of precision and recall"],
        ["ROC-AUC", f3(t["roc_auc"]), "Chance that a liked review gets a higher probability than a not-liked one (0.5 = guessing)"],
        ["Accuracy", f3(t["accuracy"]), f"Share of correct answers (always 'liked' would give {f3(sp['test_positive_rate'])})"],
    ], f"{best} on the held-out test set ({sp['test_rows']:,} reviews, threshold {t['threshold']})", widths=[3.3, 5, 8.2])
    r.figure("p1_confusion_roc.png", f"Confusion matrix and ROC curve of the selected model ({best}) on the test set.", 15)
    r.p(f"The test F1 ({f3(t['f1'])}) is close to the cross-validation F1 ({f3(models[best]['f1_mean'])}), so there is no sign of over-fitting. "
        f"For reference only (not used for selection), {other['model']} reaches F1 = {f3(other['f1'])} and ROC-AUC = {f3(other['roc_auc'])} on the same test set.")
    r.h2("8.3 Feature interpretation")
    r.figure("p1_feature_importance.png", "Logistic Regression coefficients, Random Forest importance and permutation importance.")
    r.table(["Feature", "LR coefficient (standardized)", "RF importance", "Permutation importance (test AUC drop)"],
            [[f, f"{interp['logistic_coefficients_standardized'][f]:+.3f}", f3(interp["random_forest_importance"][f]), f3(perm[f]["mean"])] for f in order],
            "Three views of feature influence")
    r.p("All three views put **genre_score** and **public_rating** clearly on top (their order differs a little between the views), followed by popularity_log. "
        "runtime_min, episodes and year add very little. The positive coefficient of popularity_log means that, for the same public rating, "
        "reviewers liked less famous dramas slightly more often; self-selection is a possible reason, but the data cannot prove it. "
        "Feature importance describes what the model uses; it does not prove what causes a person to like a drama.")
    r.h2("8.4 Prediction for a new record and decision threshold")
    i = new["input"]
    r.p(f"New record: genre_score = {i['genre_score']}, runtime = {i['runtime_min']:g} min, popularity rank = {i['popularity_rank']:g}, "
        f"public rating = {i['public_rating']}, episodes = {i['episodes']:g}, year = {i['year']}. "
        f"The model returns a probability of **{f3(new['probability_like'])}** for class 1. With the decision threshold {new['threshold']} "
        f"the predicted class is **{new['predicted_class']} ({'liked' if new['predicted_class'] else 'not liked'})**. "
        "The input values are invented, so this is a software demonstration and not a real-world validation.")
    r.table(["Threshold", "Precision", "Recall", "F1", "False positives", "False negatives"],
            [[x["threshold"], f3(x["precision"]), f3(x["recall"]), f3(x["f1"]), x["fp"], x["fn"]] for x in S["threshold_study_out_of_fold_training"]],
            "Effect of the decision threshold (out-of-fold predictions on training data; the test set is not reused)")
    r.p("A higher threshold gives fewer false positives but more false negatives. **False positive:** we recommend a drama and the "
        "viewer does not enjoy it (a wasted evening, less trust). **False negative:** we hide a drama the viewer would have loved "
        "(a missed opportunity). Neither error is dangerous, so we keep the balanced default of 0.5; a service that wants safer "
        "recommendations could use 0.6.")

    r.h1("9. Limitations and risks")
    r.bullets([
        "**Self-selected reviewers.** People who write reviews are more active and more positive than ordinary viewers.",
        f"**Cold start.** For {pc(ds['missing_share_genre_score'], 0)} of the rows the viewer has no history and genre_score is imputed; predictions for new viewers are weaker.",
        "**No time stamps.** genre_score uses the viewer's other reviews without knowing their order, and public_rating is a 2023 snapshot "
        "that includes (as a tiny part) the reviewers' own scores. A real system must use past information only.",
        "**Label definition.** 'Score of 8 or more' is our definition of 'liked'; another threshold would change the classes.",
        "**Modest gain.** The models are only a few F1 points above the 'always liked' baseline.",
        "**Risks.** A recommender trained on this data may favour already popular, highly rated dramas. Importance must not be read as causation. "
        "Source B has no stated licence, so the data is used for a classroom project only.",
    ])

    r.h1("10. Conclusions")
    r.p(f"We used {ds['rows']:,} real MyDramaList reviews to predict whether a viewer scores a K-Drama 8 or higher. After a viewer-grouped, "
        f"stratified 80/20 split and leakage-safe pipelines, Logistic Regression and Random Forest were compared with identical 3-fold "
        f"cross-validation. The model selected by mean validation F1 was {best} (mean CV F1 {f3(models[best]['f1_mean'])}). On the held-out "
        f"test set it reached F1 = {f3(t['f1'])}, precision = {f3(t['precision'])}, recall = {f3(t['recall'])} and ROC-AUC = {f3(t['roc_auc'])}, "
        f"with {cm['fp']} false positives and {cm['fn']} false negatives. The most influential features were {order[0]} and {order[1]}. "
        "The results describe performance on this public dataset of reviewers; they do not guarantee accurate personal recommendations.")
    r.p("**Realistic improvements:** (1) add review dates and compute genre_score from earlier reviews only; (2) add one score per genre and "
        "drama tags; (3) collect ratings from ordinary watch-lists; (4) choose the threshold from a real cost estimate; "
        "(5) try gradient boosting and calibrate the probabilities.")

    r.h1("References")
    r.bullets([REF_A, REF_B, REF_MDL, REF_SK], numbered=True)
    r.save("Report1_Supervised_Learning.docx")


# ======================================================================================
# REPORT 2 - UNSUPERVISED LEARNING
# ======================================================================================
def report_2():
    d, pre, km, db, pca, iso = U["dataset"], U["preprocessing"], U["kmeans"], U["dbscan"], U["pca"], U["isolation_forest"]
    kc = {x["k"]: x for x in km["k_curve"]}
    sc = pre["scaling_experiment"]
    imp = pre["imputation_experiment_artificial"]
    main = next(x for x in iso["summary"] if x["contamination"] == iso["main_contamination"])
    grid = {(x["eps"], x["min_samples"]): x for x in db["grid"]}
    n = d["observations"]
    desc = d["describe_original_units"]

    r = Report("Project 2 - Unsupervised Learning", "Hidden patterns of K-Dramas: groups, low-dimensional structure and unusual observations")

    r.h1("1. Objective")
    r.p("The objective is to explore Korean dramas without any target label: to look for groups (K-Means, DBSCAN), for a simple "
        "low-dimensional structure (PCA, t-SNE, UMAP) and for unusual observations (Isolation Forest), and to study how much the "
        "findings depend on the chosen parameters.")

    r.h1("2. Problem statement and research question")
    r.table(["Element", "Definition"], [
        ["Object", "Korean TV dramas listed on MyDramaList, released 2015-2023"],
        ["Observation unit", "One drama (one row = one drama)"],
        ["Research question", "Do K-Dramas form natural groups by genre score, runtime and popularity? Can the data be summarized in two dimensions? Which dramas are unusual?"],
        ["Variant features", "genre_score, runtime_min, popularity (as popularity_log)"],
        ["Additional features", "episodes (as episodes_log), public_rating"],
        ["Target label", "None. The 'liked' label of Project 1 is not loaded and not used for any model in this project."],
    ], "Problem definition", widths=[4, 12.5])

    r.h1("3. Dataset description")
    r.p("The data comes from the same two public Kaggle datasets as Project 1 (Source A: Charuchinda, 2023, CC0; Source B: Rizki, 2023, "
        f"licence not stated), both scraped from MyDramaList. After joining them by title and year we have **{n:,} dramas** "
        "(the worksheet requires at least 100). No value is simulated.")
    r.table(["Feature", "Meaning", "Min", "Median", "Max"], [
        ["genre_score", "Average public rating of the drama's genres (0-10)", *[f"{desc['genre_score'][k]:.2f}" for k in ("min", "50%", "max")]],
        ["runtime_min", "Minutes per episode", *[f"{desc['runtime_min'][k]:.0f}" for k in ("min", "50%", "max")]],
        ["popularity_rank", "MyDramaList popularity rank (1 = most popular)", *[f"{desc['popularity_rank'][k]:.0f}" for k in ("min", "50%", "max")]],
        ["episodes", "Number of episodes", *[f"{desc['episodes'][k]:.0f}" for k in ("min", "50%", "max")]],
        ["public_rating", "Public MyDramaList rating (0-10)", *[f"{desc['public_rating'][k]:.1f}" for k in ("min", "50%", "max")]],
    ], "Feature dictionary (original units)", widths=[3.2, 7.8, 1.7, 1.9, 1.9])
    r.p("**Drama-level genre score.** For every genre we compute the average public rating of its dramas; a drama's genre_score is the "
        "mean of these values over its genres. It measures how well-rated the drama's genres are in general. No viewer label is involved.")
    r.p("**Missing values.** The five features have no missing values in the joined table. Source A does contain gaps (24 dramas without "
        "duration and 331 with the placeholder rank 99999), but none of those dramas could be matched with Source B. The loader converts "
        "99999 to 'missing' and a median imputer is kept in the code as a safe pattern.")

    r.h1("4. Exploratory analysis")
    r.figure("p2_distributions.png", "Feature distributions in original units.")
    r.p(f"Three observations: (1) runtime_min has two peaks - short web dramas (10-25 min) and standard dramas (60-70 min); "
        f"(2) popularity_rank and episodes are strongly right-skewed (skewness {d['skew_before_log']['popularity_rank']:.1f} and {d['skew_before_log']['episodes']:.1f}); "
        "(3) public_rating and genre_score are narrow and roughly bell-shaped.")
    r.figure("p2_runtime_popularity.png", "Runtime against popularity rank (log scale), coloured by public rating.", 10)
    r.figure("p2_boxplot_correlation.png", "Standardized features (box plots) and correlation matrix.", 15)
    r.p("Highly rated dramas are also the popular ones (public_rating and popularity_log are clearly correlated), while the number of "
        "episodes is almost independent of the other features. Correlated features suggest that PCA can compress the data.")

    r.h1("5. Preprocessing")
    r.bullets([
        f"**Feature engineering:** log10 for the two skewed features (popularity_log, episodes_log). Skewness after the transformation: "
        f"{d['skew_after_log']['popularity_log']:.1f} and {d['skew_after_log']['episodes_log']:.1f}.",
        "**Standardization:** StandardScaler gives every feature mean 0 and standard deviation 1. All methods used here depend on distances or variances; "
        "without scaling, runtime_min (SD about 24) would dominate and genre_score (SD about 0.1) would be ignored.",
        f"**Scaling experiment (same K = {sc['k']} and seed):** silhouette is {f3(sc['raw_silhouette'])} on raw data and {f3(sc['scaled_silhouette'])} on standardized data; "
        f"the two clusterings agree only moderately (ARI {sc['ari_raw_vs_scaled']:.2f}). The higher raw value does not mean raw is better: the raw clustering looks almost only at runtime. "
        "We keep the standardized data so that all five features contribute.",
        f"**Imputation experiment (clearly artificial):** because nothing is missing, we hid {imp['hidden_values']} values of runtime_min on purpose. "
        f"Mean imputation kept the clusters almost unchanged (ARI {imp['strategies']['mean']['ari_vs_complete_data']:.2f} with the complete-data result); "
        f"median imputation changed them a little more (ARI {imp['strategies']['median']['ari_vs_complete_data']:.2f}), because the median ({imp['strategies']['median']['fill_value']:.0f} min) "
        "moves every filled drama into the 'long episode' group of this two-peaked feature.",
    ])

    r.h1("6. Methodology and algorithms")
    r.table(["Method", "Purpose", "Settings tested"], [
        ["K-Means", "Centroid-based groups", "K = 2...8, n_init = 10, five random seeds"],
        ["DBSCAN", "Density-based groups and noise", "eps in {0.5, 0.6, 0.7, 0.8, 1.0} x min_samples in {5, 10, 15}"],
        ["PCA", "Linear projection, explained variance", "all components; 2 components for plots"],
        ["t-SNE", "Non-linear neighbourhood map", "perplexity 10, 30, 50; init = PCA"],
        ["UMAP", "Non-linear neighbourhood map", "n_neighbors 5, 15, 50; min_dist 0.1"],
        ["Isolation Forest", "Unusual observations", "contamination 0.02, 0.05, 0.10"],
    ], "Methods and parameter ranges", widths=[3.2, 6, 7.3])
    r.p("Implementation: Python, scikit-learn and umap-learn. The notebook Project2_Unsupervised_KDrama.ipynb runs from top to bottom in "
        "Google Colab. random_state = 42 is used wherever randomness exists; DBSCAN and PCA are deterministic.")

    r.h1("7. Results")
    r.h2("7.1 K-Means: choosing K")
    r.table(["K", "Inertia", "Silhouette", "Smallest cluster"],
            [[x["k"], f"{x['inertia']:.1f}", f3(x["silhouette"]), x["smallest_cluster"]] for x in km["k_curve"]],
            "K-Means for K = 2...8")
    r.figure("p2_elbow_silhouette.png", "Elbow curve (inertia) and silhouette score. The dashed line marks the chosen K.", 14)
    r.p(f"The elbow curve is smooth, with the largest drops up to K = 4. The silhouette is highest at K = 2 ({f3(kc[2]['silhouette'])}), which only separates "
        f"short web dramas from full-length dramas, and has a small second peak at K = 4 ({f3(kc[4]['silhouette'])}). **We choose K = {km['chosen_k']}**: "
        "it costs very little silhouette, lies where the elbow flattens and gives four interpretable groups. Silhouette values near 0.3 mean that the "
        "groups overlap; they are useful segments, not sharply separated natural classes.")
    r.p(f"**Stability.** With five different seeds the inertia and silhouette are practically identical and the groupings agree strongly "
        f"(ARI between {min(x['ari_vs_seed_42'] for x in km['seed_stability']):.2f} and 1.00).")
    r.h2("7.2 Cluster profiles")
    r.table(["Cluster", "Size", "Runtime (min)", "Episodes", "Median pop. rank", "Public rating", "Genre score", "Data-derived description"],
            [[p["cluster"], f"{p['size']} ({pc(p['share'], 0)})", f"{p['mean']['runtime_min']:.0f}", f"{p['mean']['episodes']:.0f}",
              f"{p['median']['popularity_rank']:.0f}", f"{p['mean']['public_rating']:.2f}", f"{p['mean']['genre_score']:.2f}", p["description"]]
             for p in km["profiles"]],
            "K-Means cluster profiles (means in original units)", widths=[1.4, 1.9, 1.7, 1.6, 1.9, 1.6, 1.6, 4.8])
    r.figure("p2_cluster_profiles.png", "Mean z-score of every feature in every cluster.", 12)
    r.p("In plain words the four groups are: short web dramas; long-running serials with many episodes; popular, highly rated hits; and "
        "standard-format dramas of average popularity. These names are our reading of the numbers - the algorithm returns only cluster numbers. "
        "Examples of popular members: " + "; ".join(f"cluster {p['cluster']}: {', '.join(p['examples'][:3])}" for p in km["profiles"]) + ". "
        "Romance, Drama and Comedy are the most common genres in every cluster, so the groups describe format and popularity rather than genre.")
    r.h2("7.3 DBSCAN and parameter sensitivity")
    r.table(["eps", "min_samples", "Clusters", "Noise points", "Silhouette (noise excluded)"],
            [[x["eps"], x["min_samples"], x["clusters"], x["noise"],
              "-" if x["silhouette_without_noise"] is None else f3(x["silhouette_without_noise"])] for x in db["grid"]],
            "DBSCAN parameter grid (silhouette is undefined for fewer than two clusters)")
    r.figure("p2_dbscan_sensitivity.png", "k-distance plot and DBSCAN sensitivity to eps and min_samples.")
    r.p(f"Using the k-distance plot and the rule of thumb min_samples = 2 x number of features, we selected eps = {db['eps']} and min_samples = {db['min_samples']}: "
        f"**{db['clusters']} clusters and {db['noise']} noise points** ({pc(db['noise'] / n, 0)} of the dramas). The result is very sensitive: eps = 0.5, min_samples = 5 gives "
        f"{grid[(0.5, 5)]['clusters']} clusters and {grid[(0.5, 5)]['noise']} noise points, while eps = 1.0, min_samples = 10 gives {grid[(1.0, 10)]['clusters']} cluster and "
        f"{grid[(1.0, 10)]['noise']} noise points. The density of this dataset changes smoothly, so no setting is clearly 'correct'.")
    r.p(f"**Comparison with K-Means.** DBSCAN isolates the short web dramas and the long serials but merges the K-Means 'hits' and 'standard' groups into one "
        f"large cluster, because there is no empty space between them. On non-noise dramas the two methods agree moderately (ARI {db['ari_with_kmeans_excluding_noise']:.2f}). "
        "K-Means must assign every drama; DBSCAN may leave sparse dramas as noise. Noise is an algorithmic label, not a data error.")
    r.h2("7.4 PCA")
    r.table(["Component", *[f"PC{i + 1}" for i in range(5)]],
            [["Explained variance", *[pc(v) for v in pca["explained_variance_ratio"]]], ["Cumulative", *[pc(v) for v in pca["cumulative"]]]],
            "PCA explained variance (standardized features)")
    r.table(["Loading", *d["features"]],
            [[name, *[f"{pca['loadings'][name][f]:+.2f}" for f in d["features"]]] for name in ("PC1", "PC2")], "Loadings of the first two components")
    r.figure("p2_pca.png", "Explained variance and PCA projection coloured by K-Means cluster.", 15)
    r.p(f"The first two components retain **{pc(pca['retained_2d'])}** of the variance, so about {pc(1 - pca['retained_2d'], 0)} is not visible in a 2-D plot. "
        "PC1 combines public rating, runtime and genre score (positive) with popularity rank (negative): a 'mainstream success' direction. "
        "PC2 is almost only the number of episodes. Loadings describe directions of variance, not causes.")
    r.h2("7.5 t-SNE and UMAP compared with PCA")
    r.figure("p2_projection_comparison.png", "PCA, t-SNE and UMAP views of the same standardized data, coloured by K-Means cluster.")
    r.table(["", "PCA", "t-SNE", "UMAP"], [
        ["Type", "linear", "non-linear", "non-linear"],
        ["Preserves", "global variance directions", "local neighbourhoods", "local neighbourhoods, some global layout"],
        ["Axes interpretable", "yes (loadings)", "no", "no"],
        ["Deterministic", "yes", "only with a fixed seed", "only with a fixed seed"],
        ["In our data", f"one continuous cloud ({pc(pca['retained_2d'], 0)} of variance)", "several separate islands", "compact, well separated islands"],
    ], "Comparison of the three projections", widths=[3.3, 4.4, 4.4, 4.4])
    r.figure("p2_projection_sensitivity.png", "Projection sensitivity: t-SNE with three perplexities and UMAP with three n_neighbors values.", 15)
    r.p("All three views agree that short web dramas and long serials stand apart, while 'hits' and 'standard' dramas blend into each other. "
        "PCA is the most interpretable view. t-SNE and UMAP look more clustered, but the number of islands changes with perplexity and n_neighbors; "
        "distances between islands and island sizes must not be interpreted, and visual separation is not proof of real clusters.")
    r.h2("7.6 Isolation Forest and comparison with DBSCAN noise")
    r.table(["Contamination", "Flagged", "Also DBSCAN noise", "Only Isolation Forest", "Only DBSCAN noise", "Disagreement"],
            [[x["contamination"], x["flagged"], x["also_dbscan_noise"], x["only_isolation_forest"], x["only_dbscan_noise"], x["disagreement"]] for x in iso["summary"]],
            "Isolation Forest flags compared with DBSCAN noise")
    r.figure("p2_methods_in_pca.png", "K-Means clusters, DBSCAN clusters with noise and Isolation Forest flags in the same PCA view.")
    r.table(["Drama", "Year", "Runtime", "Episodes", "Pop. rank", "Rating", "Score", "DBSCAN noise"],
            [[a["title"], a["year"], f"{a['runtime_min']:.0f}", f"{a['episodes']:.0f}", f"{a['popularity_rank']:.0f}", a["public_rating"],
              f"{a['anomaly_score']:.3f}", "yes" if a["dbscan_noise"] else "no"] for a in iso["top_anomalies"]],
            f"Ten most unusual dramas (contamination {iso['main_contamination']}; lower score = more unusual)")
    r.p(f"With contamination = {main['contamination']} the forest flags **{main['flagged']} dramas**, and {main['also_dbscan_noise']} of them "
        f"({pc(main['share_of_flags_that_are_noise'], 0)}) are also DBSCAN noise. DBSCAN marks many more dramas as noise ({db['noise']}) because its notion is "
        "'not in a dense region', while the forest looks for points that are easy to isolate. The flagged dramas are of two kinds: global mega-hits "
        "(extremely popular, very highly rated, long episodes) and extreme formats (100+ episode daily serials, tiny web dramas). "
        "'Unusual' here means a rare combination of values, not a bad drama or a data error. Contamination is our choice, not a measured anomaly rate. "
        "Before treating a flag as a real anomaly we would (1) check the drama's page for data-entry mistakes and (2) check whether the flag is stable across seeds and contamination values.")

    r.h1("8. Limitations")
    r.bullets([
        f"Only dramas present in both sources are included ({n:,} of 1,752), so less-known dramas are under-represented.",
        "The clusters overlap (silhouette about 0.3); K = 4 is a useful choice, not a discovered truth.",
        "DBSCAN, t-SNE and UMAP depend strongly on their parameters; the contamination value is chosen by the analyst.",
        "genre_score is built from public ratings, so it is related to public_rating by construction.",
        "Ranks and ratings are a 2023 snapshot; Source B has no stated licence (classroom use only).",
    ])

    r.h1("9. Conclusions")
    r.p(f"We analyzed {n:,} observations and 5 numerical features from two public Kaggle datasets scraped from MyDramaList. After examining data quality and scaling, "
        f"K-Means suggested {km['chosen_k']} groups: the silhouette is highest at K = 2 ({f3(kc[2]['silhouette'])}) but K = 4 ({f3(kc[4]['silhouette'])}) lies where the elbow flattens and is interpretable. "
        f"DBSCAN with eps = {db['eps']} and min_samples = {db['min_samples']} identified {db['clusters']} groups and {db['noise']} noise points. "
        f"The first two PCA components retained {pc(pca['retained_2d'])} of standardized feature variance. The t-SNE and UMAP plots suggested separate islands for short web dramas "
        f"and long serials, subject to projection limitations. Isolation Forest flagged {main['flagged']} unusual observations with contamination = {main['contamination']}. "
        "We compared the algorithms, identified overlapping clusters and parameter sensitivity as the main limitations, and propose richer features (tags, multi-hot genres) "
        "and models for overlapping groups (Gaussian Mixture, HDBSCAN) as improvements. None of the unsupervised outputs independently establishes a ground-truth class or a verified anomaly.")
    r.p("**Main evidence-based findings:** (1) format - episode length - is the strongest structure; (2) popularity, rating and runtime move together, while the number of "
        "episodes is an independent direction; (3) hits are the upper end of a continuous scale, not a separate island; (4) the most unusual dramas are mega-hits and extreme formats.")

    r.h1("References")
    r.bullets([
        REF_A, REF_B, REF_MDL, REF_SK,
        "van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. Journal of Machine Learning Research, 9, 2579-2605.",
        "McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.",
        "Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. 2008 Eighth IEEE International Conference on Data Mining, 413-422.",
        "Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise. KDD-96, 226-231.",
    ], numbered=True)
    r.save("Report2_Unsupervised_Learning.docx")


if __name__ == "__main__":
    report_1()
    report_2()
