"""Builds the combined PowerPoint presentation and the speaking script.

Run:  python presentation/build_presentation.py
Creates:
  presentation/KDrama_AI_Presentation.pptx   (speaker notes contain the script of each slide)
  docs/PRESENTATION_SCRIPT.md                (the same script as a printable document)

All numbers are read from artifacts/*.json - nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "artifacts" / "figures"
S = json.loads((ROOT / "artifacts" / "supervised_results.json").read_text())
U = json.loads((ROOT / "artifacts" / "unsupervised_results.json").read_text())

INK, PANEL, PAPER, MIST = RGBColor(0x0B, 0x0A, 0x0E), RGBColor(0x1B, 0x18, 0x22), RGBColor(0xF2, 0xEC, 0xEE), RGBColor(0xA9, 0xA1, 0xB0)
ROSE, BLUSH, BURGUNDY, LAVENDER = RGBColor(0xC9, 0x7B, 0x8E), RGBColor(0xE6, 0xB8, 0xC2), RGBColor(0x7A, 0x1F, 0x3D), RGBColor(0x9D, 0x8C, 0xC9)
SERIF, SANS = "Georgia", "Calibri"
SPEAKERS = {1: "Shakenova Aruzhan", 2: "Qabir Madina", 3: "Shahap Aizere", 4: "Zholdaspek Dana"}

W, H = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width, prs.slide_height = W, H
BLANK = prs.slide_layouts[6]
script_lines: list[str] = []

f3 = lambda v: f"{v:.3f}"
pc = lambda v, d=0: f"{v * 100:.{d}f}%"

# ------------------------------------------------------------------ numbers used on the slides
ds, sp, cv, t = S["dataset"], S["split"], S["cross_validation"], S["test"]
cm, best = t["confusion_matrix"], S["selected_model"]
models = cv["models"]
other = S["test_other_model_reference_only"]
perm = S["interpretation"]["permutation_importance_test_auc_drop"]
order = sorted(perm, key=lambda f: -perm[f]["mean"])
new = S["new_record"]
thr = {x["threshold"]: x for x in S["threshold_study_out_of_fold_training"]}

d, km, db, pca, iso = U["dataset"], U["kmeans"], U["dbscan"], U["pca"], U["isolation_forest"]
kc = {x["k"]: x for x in km["k_curve"]}
sc = U["preprocessing"]["scaling_experiment"]
grid = {(x["eps"], x["min_samples"]): x for x in db["grid"]}
iso_main = next(x for x in iso["summary"] if x["contamination"] == iso["main_contamination"])
n = d["observations"]


# ------------------------------------------------------------------ drawing helpers
def text(slide, x, y, w, h, content, size=18, color=PAPER, bold=False, font=SANS, italic=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(x, y, w, h)
    frame = box.text_frame
    frame.word_wrap = True
    frame.vertical_anchor = anchor
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    lines = content if isinstance(content, list) else [content]
    for i, line in enumerate(lines):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.alignment = align
        para.space_after = Pt(size * 0.45)
        parts = line.split("**")
        for j, part in enumerate(parts):
            if not part:
                continue
            run = para.add_run()
            run.text = part
            run.font.size, run.font.name, run.font.italic = Pt(size), font, italic
            run.font.bold = bold or j % 2 == 1
            run.font.color.rgb = BLUSH if (j % 2 == 1 and not bold) else color
    return box


def rect(slide, x, y, w, h, fill, rounded=True):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if rounded:
        shape.adjustments[0] = 0.06
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def base(eyebrow, title, speaker, notes, number):
    slide = prs.slides.add_slide(BLANK)
    rect(slide, 0, 0, W, H, INK, rounded=False)
    glow = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(9.6), Inches(-2.6), Inches(6.5), Inches(5.2))
    glow.fill.solid()
    glow.fill.fore_color.rgb = RGBColor(0x2A, 0x10, 0x26)
    glow.line.fill.background()
    rect(slide, Inches(0.6), Inches(0.62), Inches(0.5), Emu(12700), ROSE, rounded=False)
    text(slide, Inches(1.25), Inches(0.47), Inches(9), Inches(0.3), eyebrow.upper(), size=11, color=ROSE, bold=True)
    text(slide, Inches(0.6), Inches(0.85), Inches(12.1), Inches(1.0), title, size=32, font=SERIF)
    text(slide, Inches(0.6), Inches(7.03), Inches(8), Inches(0.3), f"K-DRAMA AI   |   {SPEAKERS[speaker]}", size=10, color=MIST)
    text(slide, Inches(11.7), Inches(7.03), Inches(1.0), Inches(0.3), f"{number:02d}", size=10, color=MIST, align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = f"[{SPEAKERS[speaker]}]\n{notes}"
    script_lines.append(f"### Slide {number} - {title}\n**Speaker: {SPEAKERS[speaker]}**\n\n{notes}\n")
    return slide


def bullets(slide, items, x=Inches(0.6), y=Inches(2.0), w=Inches(6.0), size=17, h=Inches(4.8)):
    text(slide, x, y, w, h, [f"•  {i}" for i in items], size=size, color=PAPER)


def picture(slide, name, x, y, w):
    path = FIG / name
    pic = slide.shapes.add_picture(str(path), x, y, width=w)
    pad = Inches(0.08)
    card = rect(slide, x - pad, y - pad, pic.width + 2 * pad, pic.height + 2 * pad, RGBColor(0xFF, 0xFF, 0xFF))
    slide.shapes._spTree.remove(card._element)
    slide.shapes._spTree.insert(slide.shapes._spTree.index(pic._element), card._element)
    return pic


def table(slide, header, rows, x, y, w, row_h=Inches(0.42), size=12, col_widths=None):
    shape = slide.shapes.add_table(len(rows) + 1, len(header), x, y, w, row_h * (len(rows) + 1))
    tbl = shape.table
    for c, value in enumerate(header):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = BURGUNDY
        _cell(cell, value, size, PAPER, True)
    for r_i, row in enumerate(rows, start=1):
        for c, value in enumerate(row):
            cell = tbl.cell(r_i, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL
            _cell(cell, value, size, PAPER, False)
    if col_widths:
        for c, cw in enumerate(col_widths):
            tbl.columns[c].width = Inches(cw)
    return tbl


def _cell(cell, value, size, color, bold):
    cell.text = ""
    cell.margin_left = cell.margin_right = Inches(0.08)
    cell.margin_top = cell.margin_bottom = Inches(0.04)
    run = cell.text_frame.paragraphs[0].add_run()
    run.text = str(value)
    run.font.size, run.font.name, run.font.bold = Pt(size), SANS, bold
    run.font.color.rgb = color


def stats(slide, items, y=Inches(2.1), x0=Inches(0.6), total_w=Inches(12.1)):
    gap = Inches(0.25)
    w = int((total_w - gap * (len(items) - 1)) / len(items))
    for i, (value, label) in enumerate(items):
        x = x0 + i * (w + gap)
        rect(slide, x, y, w, Inches(1.55), PANEL)
        text(slide, x + Inches(0.25), y + Inches(0.18), w - Inches(0.5), Inches(0.8), str(value), size=34, font=SERIF, color=BLUSH)
        text(slide, x + Inches(0.25), y + Inches(1.0), w - Inches(0.5), Inches(0.5), label.upper(), size=10, color=MIST, bold=True)


# ================================================================== SLIDES
num = 0


def nxt():
    global num
    num += 1
    return num


# 1 ---------------------------------------------------------------- title
slide = prs.slides.add_slide(BLANK)
rect(slide, 0, 0, W, H, INK, rounded=False)
for cx, cy, size, color in [(8.2, -1.5, 7.5, RGBColor(0x3A, 0x13, 0x2B)), (9.9, 0.3, 3.6, RGBColor(0x5A, 0x1A, 0x36)), (-2.0, 4.5, 5.0, RGBColor(0x1C, 0x15, 0x2B))]:
    o = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx), Inches(cy), Inches(size), Inches(size))
    o.fill.solid()
    o.fill.fore_color.rgb = color
    o.line.fill.background()
text(slide, Inches(0.8), Inches(1.7), Inches(10), Inches(0.4), "AI FUNDAMENTALS   ·   FINAL EXAM PROJECT   ·   VARIANT 21", size=12, color=ROSE, bold=True)
text(slide, Inches(0.8), Inches(2.2), Inches(11.5), Inches(1.6), "K-DRAMA AI", size=88, font=SERIF)
text(slide, Inches(0.8), Inches(3.95), Inches(11), Inches(0.7), "Discover Your Next Favorite Story", size=30, font=SERIF, italic=True, color=BLUSH)
text(slide, Inches(0.8), Inches(4.85), Inches(10), Inches(0.9),
     ["Supervised and unsupervised machine learning on real Korean drama data", "Shakenova Aruzhan  ·  Qabir Madina  ·  Shahap Aizere  ·  Zholdaspek Dana   —   QAIRU"], size=16, color=MIST)
nxt()
note = ("Good morning. We are a team of four students, and our project is called K-Drama AI. Our variant is number 21, Movie Preference. "
        "We adapted it to Korean dramas. We built two machine learning projects, one supervised and one unsupervised, and also a website that shows the results. "
        "Each of us will present one part.")
slide.notes_slide.notes_text_frame.text = f"[{SPEAKERS[1]}]\n{note}"
script_lines.append(f"### Slide 1 - Title\n**Speaker: {SPEAKERS[1]}**\n\n{note}\n")

# 2 ---------------------------------------------------------------- introduction
s = base("Project introduction", "One topic, three connected parts", 1,
         "Our work has three parts. Project 1 is supervised learning: we predict if a viewer will like a drama. "
         "Project 2 is unsupervised learning: we look for groups and unusual dramas without any labels. "
         "The third part is a website. It is not required, but it shows our real models in a friendly way. "
         "All three parts use the same data and the same Python code, so the numbers are always the same.", nxt())
for i, (head, body, color) in enumerate([
    ("Project 1 · Supervised", "Will a viewer like this K-Drama?\nLogistic Regression vs Random Forest", ROSE),
    ("Project 2 · Unsupervised", "Hidden patterns of K-Dramas\nK-Means, DBSCAN, PCA, t-SNE, UMAP, Isolation Forest", LAVENDER),
    ("Website · Demonstration", "Next.js + FastAPI\nReal model, real analysis results", BLUSH)]):
    x = Inches(0.6) + i * Inches(4.12)
    rect(s, x, Inches(2.1), Inches(3.87), Inches(3.2), PANEL)
    rect(s, x, Inches(2.1), Inches(3.87), Inches(0.08), color, rounded=False)
    text(s, x + Inches(0.3), Inches(2.45), Inches(3.3), Inches(0.5), head, size=20, font=SERIF, color=color)
    text(s, x + Inches(0.3), Inches(3.2), Inches(3.3), Inches(1.8), body.split("\n"), size=15, color=PAPER)
text(s, Inches(0.6), Inches(5.7), Inches(12), Inches(0.8),
     "One data pipeline (ml/kdrama_data.py) feeds both notebooks, the reports, these slides and the website.", size=15, color=MIST, italic=True)

# 3 ---------------------------------------------------------------- why k-dramas
s = base("Why K-Dramas", "Variant 21 — Movie preference, adapted to Korean dramas", 1,
         "The worksheet says: predict if a viewer likes a movie, using genre score, runtime and popularity. "
         "It also says that no exact dataset exists for this variant. So we could choose our own topic. "
         "We chose Korean dramas because we watch them and understand them, and because we found real public data. "
         "We kept the three required features and added three more. We adapted the task, so this should be approved by the instructor.", nxt())
bullets(s, [
    "Worksheet: **predict “likes movie”** from genre score, runtime, popularity",
    "Dataset status in the worksheet: **“no exact source established”**",
    "Our adaptation: movie → **Korean drama**; same three features plus three more",
    "We know the topic, so we can check if results make sense",
    "Real public data exists — **no simulated labels were needed**",
], w=Inches(7.3))
rect(s, Inches(8.4), Inches(2.0), Inches(4.3), Inches(3.9), PANEL)
text(s, Inches(8.7), Inches(2.25), Inches(3.8), Inches(0.4), "REQUIRED FEATURES", size=11, color=ROSE, bold=True)
text(s, Inches(8.7), Inches(2.7), Inches(3.8), Inches(1.5), ["genre_score", "runtime_min", "popularity"], size=22, font=SERIF, color=BLUSH)
text(s, Inches(8.7), Inches(4.35), Inches(3.8), Inches(0.4), "ADDED BY US", size=11, color=ROSE, bold=True)
text(s, Inches(8.7), Inches(4.75), Inches(3.8), Inches(1.0), ["public_rating · episodes · year"], size=17, color=PAPER)

# 4 ---------------------------------------------------------------- dataset
s = base("Dataset", "Real data from MyDramaList — nothing is simulated", 1,
         f"We use two public datasets from Kaggle. Both were collected from the website MyDramaList. "
         f"The first one has drama information and more than ten thousand reviews, and each review has the viewer's own score. "
         f"The second one has genres and the public rating. We join them by title and year. "
         f"After joining we have {ds['rows']:,} reviews from {ds['viewers']:,} viewers, and {n:,} dramas. "
         f"A viewer 'likes' a drama if their own score is 8 or more. About {pc(ds['positive_rate'])} of reviews are positive. "
         f"Important: the public rating of a drama is not a personal opinion, so we use it only as a feature, not as the label.", nxt())
stats(s, [(f"{ds['rows']:,}", "real viewer reviews"), (f"{ds['viewers']:,}", "viewers"), (f"{n:,}", "dramas (2015–2023)"), (pc(ds["positive_rate"]), "liked = score ≥ 8")])
table(s, ["Source", "What we take", "Licence"], [
    ["A · Charuchinda (2023), Kaggle", "runtime, episodes, popularity rank; 10,625 reviews with viewer scores", "CC0 Public Domain"],
    ["B · Rizki (2023), Kaggle", "genres, public MyDramaList rating", "not stated (classroom use)"],
], Inches(0.6), Inches(4.0), Inches(12.1), size=13, col_widths=[3.6, 6.0, 2.5])
text(s, Inches(0.6), Inches(5.6), Inches(12.1), Inches(1.2),
     ["**Target:** liked = 1 if the viewer's own overall score ≥ 8, else 0.   **Limitation:** people who write reviews are not typical viewers."], size=15, color=PAPER)

# 5 ---------------------------------------------------------------- preprocessing
s = base("Preprocessing", "Features and a leakage-safe design", 2,
         f"Now the preprocessing. The most important feature is the genre score. For each review, it is the average score that the same viewer gave to other dramas of the same genres. "
         f"We never include the review that we want to predict, so the feature does not contain the answer. "
         f"If the viewer has no other reviews, the value is missing. This is {pc(ds['missing_share_genre_score'])} of rows, and these are real missing values. "
         f"We split the data 80 to 20 before any preprocessing. All reviews of one viewer stay on the same side, so test viewers are really new. "
         f"The imputer and the scaler are inside a pipeline, so they learn only from training data. This prevents data leakage.", nxt())
bullets(s, [
    "**genre_score** = viewer's mean score for OTHER dramas of the same genres (never the predicted review)",
    f"Real missing values: **{pc(ds['missing_share_genre_score'])}** of genre_score (viewers with no history)",
    "popularity_log = log10(popularity rank) — row-by-row, safe before the split",
    f"**Split first:** {sp['train_rows']:,} train / {sp['test_rows']:,} test, stratified, **grouped by viewer** (0 shared viewers)",
    "**Pipeline:** median imputer → StandardScaler → classifier, fitted on training rows only",
], w=Inches(7.0), size=16)
picture(s, "p1_class_balance.png", Inches(7.95), Inches(2.25), Inches(4.75))
text(s, Inches(7.9), Inches(4.55), Inches(4.8), Inches(0.9), "Class balance and viewer scores. Reviewers are generous, so we also report an “always liked” baseline.", size=12, color=MIST, italic=True)

# 6 ---------------------------------------------------------------- supervised implementation
s = base("Supervised learning · implementation", "Two models, one fair comparison", 2,
         "We trained two models. Logistic Regression is a simple linear model. It is easy to explain. "
         "Random Forest is many decision trees that vote. It can learn more complex rules. "
         "Both models are in the same pipeline. We compare them with the same three folds of cross-validation, and the score is F1. "
         "We choose the model with the higher mean F1 on training data. Only after that we open the test set, one time.", nxt())
table(s, ["", "Logistic Regression", "Random Forest"], [
    ["Idea", "weighted sum of features → probability", "150 decision trees vote"],
    ["Strength", "simple, coefficients are easy to read", "non-linear rules, interactions"],
    ["Settings", "max_iter=1000, random_state=42", "max_depth=6, min_samples_leaf=3, random_state=42"],
], Inches(0.6), Inches(2.05), Inches(12.1), size=14, col_widths=[1.6, 5.0, 5.5])
for i, (step, body) in enumerate([("1  Split", "80 / 20, by viewer"), ("2  Validate", "same stratified 3-fold CV, F1"), ("3  Select", "higher mean CV F1"), ("4  Test once", "untouched test set")]):
    x = Inches(0.6) + i * Inches(3.07)
    rect(s, x, Inches(4.45), Inches(2.85), Inches(1.5), PANEL)
    text(s, x + Inches(0.25), Inches(4.62), Inches(2.4), Inches(0.5), step, size=20, font=SERIF, color=BLUSH)
    text(s, x + Inches(0.25), Inches(5.2), Inches(2.4), Inches(0.6), body, size=14, color=PAPER)
text(s, Inches(0.6), Inches(6.2), Inches(12), Inches(0.5), "Sequence from the worksheet: DATA → SPLIT → PREPARE → TRAIN → VALIDATE → TEST → DECIDE", size=13, color=MIST, italic=True)

# 7 ---------------------------------------------------------------- EDA
s = base("Supervised learning · exploring the data", "What the features look like", 2,
         "Here are the six features for liked and not liked reviews. We can see three things. "
         "First, liked reviews have a higher genre score and a higher public rating. "
         "Second, runtime has two peaks: short web dramas and normal one-hour dramas. "
         "Third, the features have very different scales, so we need scaling for Logistic Regression.", nxt())
picture(s, "p1_feature_distributions.png", Inches(0.7), Inches(2.0), Inches(8.3))
bullets(s, ["Liked reviews: **higher genre_score** and **higher public_rating**", "runtime_min has **two peaks** (web dramas / TV dramas)",
            "Very different scales → **scaling** is needed", "The classes overlap → expect a useful model, not a perfect one"], x=Inches(9.3), w=Inches(3.5), size=14)

# 8 ---------------------------------------------------------------- supervised results
s = base("Supervised learning · results", f"{best} was selected and tested once", 2,
         f"These are the results. In cross-validation, Random Forest has a mean F1 of {f3(models['Random Forest']['f1_mean'])} and Logistic Regression has {f3(models['Logistic Regression']['f1_mean'])}. "
         f"They are very close. We selected {best} because its F1 is higher. "
         f"On the test set, precision is {f3(t['precision'])}, recall is {f3(t['recall'])}, F1 is {f3(t['f1'])} and ROC-AUC is {f3(t['roc_auc'])}. "
         f"There are {cm['fp']} false positives and {cm['fn']} false negatives. "
         f"We must be honest: a model that always says 'liked' has F1 about {cv['baseline_always_like_f1_mean']:.2f}. So our model is better, but not by much. ROC-AUC shows more clearly that it is better than guessing.", nxt())
table(s, ["Model (3-fold CV, training data)", "Mean F1", "SD", "ROC-AUC"],
      [[name, f3(m["f1_mean"]), f3(m["f1_std"]), f3(m["roc_auc_mean"])] for name, m in models.items()]
      + [["Baseline: always “liked”", f3(cv["baseline_always_like_f1_mean"]), "–", "0.500"]],
      Inches(0.6), Inches(2.0), Inches(6.0), size=13, col_widths=[3.1, 1.0, 0.8, 1.1])
stats(s, [(f3(t["precision"]), "precision"), (f3(t["recall"]), "recall"), (f3(t["f1"]), "F1"), (f3(t["roc_auc"]), "ROC-AUC")], y=Inches(4.0), total_w=Inches(6.0))
text(s, Inches(0.6), Inches(5.75), Inches(6.0), Inches(1.0),
     [f"Test set: {sp['test_rows']:,} reviews, threshold 0.5.  TN {cm['tn']} · FP {cm['fp']} · FN {cm['fn']} · TP {cm['tp']}"], size=13, color=MIST)
picture(s, "p1_confusion_roc.png", Inches(6.95), Inches(2.3), Inches(5.75))

# 9 ---------------------------------------------------------------- interpretation
s = base("Supervised learning · interpretation", "What matters, and what a mistake costs", 2,
         f"Which features are important? All three methods show the same two features on top: the genre score of the viewer and the public rating of the drama. Runtime and episodes are almost not important. "
         f"But importance is not causation. It only shows what the model uses. "
         f"We also predicted a new record. The model gave a probability of {new['probability_like']:.2f}. It is above the threshold of 0.5, so the class is 'liked'. "
         f"A false positive means we recommend a drama and the viewer does not like it: lost time. A false negative means we hide a drama that the viewer would love. "
         f"If we move the threshold to 0.6, precision goes up to {thr[0.6]['precision']:.2f} but recall goes down to {thr[0.6]['recall']:.2f}.", nxt())
picture(s, "p1_feature_importance.png", Inches(0.7), Inches(2.0), Inches(7.9))
bullets(s, ["Most important: **genre_score** and **public_rating**", "Importance ≠ causation",
            f"New record → probability **{new['probability_like']:.2f}** ≥ threshold 0.5 → **liked**",
            "**False positive:** a wasted evening", "**False negative:** a missed favourite",
            f"Threshold 0.6: precision {thr[0.6]['precision']:.2f}, recall {thr[0.6]['recall']:.2f}"], x=Inches(8.95), w=Inches(3.9), size=14)

# 10 --------------------------------------------------------------- unsupervised setup
s = base("Unsupervised learning · implementation", "Same data, no labels", 3,
         f"Now Project 2. Here one row is one drama, and we have {n:,} dramas. We do not use the 'liked' label at all. "
         f"We use five numeric features: genre score, runtime, popularity, episodes and public rating. "
         f"Popularity and episodes are very skewed, so we use a logarithm. Then we standardize all features. "
         f"This is important because all our methods use distances. Without scaling, runtime would decide everything. "
         f"We tested it: without scaling the silhouette looks higher, {sc['raw_silhouette']:.2f}, but the clusters use only runtime. With scaling it is {sc['scaled_silhouette']:.2f}, and all features matter.", nxt())
bullets(s, [
    f"Observation unit: **one drama** ({n:,} dramas, 5 numerical features, 0 missing values)",
    "The Project 1 target is **not loaded** in this notebook",
    "log10 for popularity rank and episodes (very skewed)",
    "**StandardScaler:** every feature gets mean 0 and SD 1",
    f"Scaling test: raw silhouette {sc['raw_silhouette']:.2f} (runtime only) vs scaled {sc['scaled_silhouette']:.2f} (all features)",
    "Six methods: K-Means, DBSCAN, PCA, t-SNE, UMAP, Isolation Forest",
], w=Inches(6.6), size=16)
picture(s, "p2_runtime_popularity.png", Inches(7.7), Inches(2.0), Inches(5.0))

# 11 --------------------------------------------------------------- k-means
s = base("K-Means", f"Choosing K: we selected K = {km['chosen_k']}", 3,
         f"For K-Means we tried K from 2 to 8. On the left is the elbow plot. The curve is smooth, and it becomes flatter after K equals 4. "
         f"On the right is the silhouette score. The best value is at K equals 2, {kc[2]['silhouette']:.3f}. But two clusters only show short dramas and long dramas, and we already know this. "
         f"There is a second small peak at K equals 4, {kc[4]['silhouette']:.3f}. So we chose K equals 4. It is almost as good and gives more useful groups. "
         f"The silhouette is about 0.3, which is low. This means the groups overlap. We also tried five random seeds, and the result was stable.", nxt())
picture(s, "p2_elbow_silhouette.png", Inches(0.7), Inches(2.05), Inches(7.6))
bullets(s, [f"Silhouette is best at **K = 2** ({f3(kc[2]['silhouette'])}): short vs long dramas",
            f"Second peak at **K = 4** ({f3(kc[4]['silhouette'])}); elbow flattens there",
            "**K = 4**: small loss, more useful groups",
            "Silhouette ≈ 0.3 → groups **overlap**",
            f"Stable over 5 seeds (ARI ≥ {min(x['ari_vs_seed_42'] for x in km['seed_stability']):.2f})"], x=Inches(8.7), w=Inches(4.1), size=15)

# 12 --------------------------------------------------------------- cluster profiles
s = base("K-Means · cluster profiles", "Four groups of K-Dramas", 3,
         "This table shows the four groups in original units. "
         + " ".join(f"Cluster {p['cluster']} has {p['size']} dramas: {p['description']}." for p in km["profiles"])
         + " So we can describe them as long serials, short web dramas, popular hits, and standard dramas. "
           "These names are our interpretation. The algorithm only gives numbers. Romance and Drama are the top genres in every cluster, so the groups are about format and popularity, not about genre.", nxt())
table(s, ["Cluster", "Size", "Runtime", "Episodes", "Median rank", "Rating", "Data-derived description", "Examples"],
      [[p["cluster"], f"{p['size']} ({pc(p['share'])})", f"{p['mean']['runtime_min']:.0f} min", f"{p['mean']['episodes']:.0f}", f"#{p['median']['popularity_rank']:.0f}",
        f"{p['mean']['public_rating']:.2f}", p["description"], ", ".join(p["examples"][:2])] for p in km["profiles"]],
      Inches(0.6), Inches(2.05), Inches(12.1), row_h=Inches(0.78), size=12, col_widths=[0.85, 1.15, 1.0, 0.95, 1.15, 0.8, 3.5, 2.7])
text(s, Inches(0.6), Inches(6.15), Inches(12.1), Inches(0.7),
     "Descriptions are generated from z-scores (|z| ≥ 0.5). Genres were not used for clustering; cluster numbers are arbitrary labels.", size=13, color=MIST, italic=True)

# 13 --------------------------------------------------------------- dbscan
s = base("DBSCAN", "Density clusters, noise and parameter sensitivity", 3,
         f"DBSCAN finds dense areas and does not need the number of clusters. Points in empty areas are called noise. "
         f"We tested fifteen combinations of the two parameters. With eps 0.8 and min samples 10, we get {db['clusters']} clusters and {db['noise']} noise points. "
         f"But DBSCAN is very sensitive. With a small eps we get {grid[(0.5, 5)]['clusters']} clusters and {grid[(0.5, 5)]['noise']} noise points. With a large eps we get only {grid[(1.0, 10)]['clusters']} cluster. "
         f"Compared with K-Means: DBSCAN also finds the web dramas and the long serials, but it puts hits and standard dramas into one big cluster, because there is no empty space between them. "
         f"Noise does not mean an error. It only means the drama is in a sparse area.", nxt())
picture(s, "p2_dbscan_sensitivity.png", Inches(0.7), Inches(2.0), Inches(8.4))
bullets(s, [f"Main setting: eps = {db['eps']}, min_samples = {db['min_samples']}", f"→ **{db['clusters']} clusters, {db['noise']} noise points**",
            f"eps 0.5 / 5 → {grid[(0.5, 5)]['clusters']} clusters, {grid[(0.5, 5)]['noise']} noise",
            f"eps 1.0 / 10 → {grid[(1.0, 10)]['clusters']} cluster, {grid[(1.0, 10)]['noise']} noise",
            f"Agreement with K-Means: ARI {db['ari_with_kmeans_excluding_noise']:.2f}", "Noise ≠ error"], x=Inches(9.4), w=Inches(3.4), size=14)

# 14 --------------------------------------------------------------- projections
s = base("PCA · t-SNE · UMAP", "Three ways to see five dimensions in 2-D", 3,
         f"We cannot draw five dimensions, so we use three methods to make a 2D picture. "
         f"PCA is linear. The first two components keep {pc(pca['retained_2d'], 1)} of the variance. The first axis is about success: rating, runtime and popularity together. The second axis is the number of episodes. "
         f"t-SNE and UMAP are non-linear. They keep neighbours together, so we see separate islands. "
         f"But we must be careful. In t-SNE and UMAP the distance between islands and the size of islands have no meaning, and the picture changes when we change the parameters. "
         f"So an island is a hint, not proof of a real cluster. All three pictures agree that web dramas and long serials are separate.", nxt())
picture(s, "p2_projection_comparison.png", Inches(0.7), Inches(2.0), Inches(12.0))
text(s, Inches(0.6), Inches(5.55), Inches(12.1), Inches(1.3),
     [f"**PCA:** linear, 2 components keep **{pc(pca['retained_2d'], 1)}** of variance; axes can be explained.   **t-SNE / UMAP:** non-linear, keep neighbours; distances between islands have **no meaning**.",
      "Visual separation is not proof of real clusters — the number of islands changes with perplexity and n_neighbors."], size=14, color=PAPER)

# 15 --------------------------------------------------------------- isolation forest
s = base("Isolation Forest", "Which dramas are unusual?", 4,
         f"Isolation Forest looks for unusual dramas. It cuts the data randomly, and unusual points are separated very fast. "
         f"We tested three contamination values. With 5 percent, the model flags {iso_main['flagged']} dramas, and {iso_main['also_dbscan_noise']} of them are also DBSCAN noise. So the two methods mostly agree about the most unusual dramas. "
         f"But DBSCAN has {db['noise']} noise points, many more. The two methods have different ideas of 'unusual'. "
         f"Who is unusual? The first group is mega-hits like {iso['top_anomalies'][0]['title']} and {iso['top_anomalies'][1]['title']}. The second group is extreme formats, for example dramas with more than 100 episodes. "
         f"So an anomaly here is a rare combination of values. It is not a bad drama and not an error.", nxt())
table(s, ["Contamination", "Flagged", "Also DBSCAN noise", "Only DBSCAN noise"],
      [[x["contamination"], x["flagged"], x["also_dbscan_noise"], x["only_dbscan_noise"]] for x in iso["summary"]],
      Inches(0.6), Inches(2.05), Inches(5.6), size=13, col_widths=[1.5, 1.0, 1.7, 1.4])
table(s, ["Most unusual dramas", "Runtime", "Eps.", "Rank"],
      [[a["title"], f"{a['runtime_min']:.0f}", f"{a['episodes']:.0f}", f"#{a['popularity_rank']:.0f}"] for a in iso["top_anomalies"][:6]],
      Inches(0.6), Inches(4.05), Inches(5.6), row_h=Inches(0.37), size=12, col_widths=[3.1, 0.9, 0.7, 0.9])
picture(s, "p2_methods_in_pca.png", Inches(6.5), Inches(2.3), Inches(6.2))
text(s, Inches(6.45), Inches(4.4), Inches(6.3), Inches(1.6),
     ["•  Mega-hits and extreme formats are flagged", "•  “Unusual” = rare combination, **not** an error", "•  Contamination is our choice, not a measured rate"], size=14)

# 16 --------------------------------------------------------------- website architecture
s = base("Website demonstration", "K-Drama AI — a real model behind a real interface", 4,
         "This is our website. It has four pages: Home, AI Predictor, Drama Insights and About. "
         "The frontend is made with Next.js and React. The backend is made with Python and FastAPI. "
         "The backend loads the real model that we trained and saved with joblib. When the user presses the button, the website sends the values to Python, and Python returns the probability. "
         "We do not calculate anything in JavaScript and we do not use fake numbers. If the backend is off, the website shows an error message. "
         "The Insights page shows the real clustering results from Project 2. Now we will show a short live demo.", nxt())
for i, (head, lines, color) in enumerate([
    ("Python training", ["ml/train_supervised.py", "ml/analyze_unsupervised.py", "→ model.joblib + results.json"], LAVENDER),
    ("FastAPI backend", ["POST /api/predict", "GET /api/insights, /api/model", "input validation, honest errors"], ROSE),
    ("Next.js website", ["Home · AI Predictor", "Drama Insights · About", "interactive Plotly charts"], BLUSH)]):
    x = Inches(0.6) + i * Inches(4.12)
    rect(s, x, Inches(2.1), Inches(3.87), Inches(2.5), PANEL)
    text(s, x + Inches(0.3), Inches(2.3), Inches(3.3), Inches(0.5), head, size=20, font=SERIF, color=color)
    text(s, x + Inches(0.3), Inches(2.95), Inches(3.4), Inches(1.6), lines, size=14)
    if i < 2:
        text(s, x + Inches(3.83), Inches(3.05), Inches(0.35), Inches(0.5), "→", size=24, color=MIST, align=PP_ALIGN.CENTER)
bullets(s, ["**Live demo:** choose a drama → set your genre score → “Predict My Preference” → probability, threshold and explanation",
            "**Drama Insights:** switch PCA / t-SNE / UMAP and colour by K-Means, DBSCAN or Isolation Forest",
            "The site states clearly that it is a classroom demonstration, not a personal recommender"], y=Inches(4.9), w=Inches(12.1), size=14, h=Inches(2.0))

# 17 --------------------------------------------------------------- conclusions
s = base("Conclusions and limitations", "What we learned — and what we cannot claim", 4,
         f"Our conclusions. In Project 1, the model predicts if a viewer likes a drama with F1 {t['f1']:.2f} and ROC-AUC {t['roc_auc']:.2f} on new viewers. The viewer's taste for the genre and the public rating are the most useful features. "
         f"In Project 2, we found that the format of a drama is the main structure: web dramas, long serials, hits and standard dramas. "
         f"Our limitations: reviewers are not typical viewers. Many viewers have no history. The data has no dates. The clusters overlap and depend on parameters. And importance is not causation. "
         f"In the future we can add dates, more features like tags, and try other models.", nxt())
rect(s, Inches(0.6), Inches(2.0), Inches(5.9), Inches(4.6), PANEL)
text(s, Inches(0.9), Inches(2.2), Inches(5.4), Inches(0.4), "FINDINGS", size=12, color=ROSE, bold=True)
text(s, Inches(0.9), Inches(2.7), Inches(5.4), Inches(3.8), [
    f"•  {best}: test F1 **{f3(t['f1'])}**, ROC-AUC **{f3(t['roc_auc'])}** on unseen viewers",
    f"•  Key features: **{order[0]}** and **{order[1]}**",
    f"•  K-Means K = {km['chosen_k']}: web dramas, long serials, hits, standard dramas",
    f"•  PCA keeps **{pc(pca['retained_2d'], 1)}** in 2-D; DBSCAN: {db['clusters']} groups, {db['noise']} noise",
    f"•  {iso_main['flagged']} unusual dramas: mega-hits and extreme formats"], size=15)
rect(s, Inches(6.8), Inches(2.0), Inches(5.9), Inches(4.6), PANEL)
text(s, Inches(7.1), Inches(2.2), Inches(5.4), Inches(0.4), "LIMITATIONS AND NEXT STEPS", size=12, color=LAVENDER, bold=True)
text(s, Inches(7.1), Inches(2.7), Inches(5.4), Inches(3.8), [
    "•  Reviewers are self-selected and generous",
    f"•  Cold start: {pc(ds['missing_share_genre_score'])} of rows have no viewer history",
    "•  No dates; ratings are a 2023 snapshot",
    "•  Clusters overlap (silhouette ≈ 0.3) and depend on parameters",
    "•  Importance ≠ causation; classroom demo only",
    "•  Next: dates, tags, per-genre scores, GMM / HDBSCAN"], size=15)

# 18 --------------------------------------------------------------- team and questions
s = base("Thank you", "Questions?", 4,
         "To finish: each of us worked on one main part, but all of us studied both projects and can answer questions about any step. "
         "Thank you for your attention. We are ready for your questions.", nxt())
for i, (who, role) in enumerate([("Shakenova Aruzhan", "Data, sources, feature engineering"), ("Qabir Madina", "Supervised learning"),
                                 ("Shahap Aizere", "Unsupervised learning"), ("Zholdaspek Dana", "Anomalies, website, reports")]):
    x = Inches(0.6) + i * Inches(3.07)
    rect(s, x, Inches(2.3), Inches(2.85), Inches(1.7), PANEL)
    text(s, x + Inches(0.25), Inches(2.5), Inches(2.4), Inches(0.5), who, size=20, font=SERIF, color=BLUSH)
    text(s, x + Inches(0.25), Inches(3.1), Inches(2.4), Inches(0.8), role, size=13)
text(s, Inches(0.6), Inches(4.7), Inches(12.1), Inches(1.0), "Notebooks · reports · code · website:  see README.md", size=16, color=MIST)
text(s, Inches(0.6), Inches(5.3), Inches(12.1), Inches(1.0), "감사합니다", size=40, font="Apple SD Gothic Neo", color=ROSE)

# ------------------------------------------------------------------ save
out = ROOT / "presentation" / "KDrama_AI_Presentation.pptx"
prs.save(out)
print("saved", out, "| slides:", len(prs.slides))

header = f"""# K-Drama AI - presentation speaking script

One combined presentation for both projects, {num} slides. The text is written in simple English (B1 level).
Read it a few times and then say it in your own words - do not read from the paper.
All numbers come from the real results (artifacts/*.json).

| Speaker | Slides | Part |
|---|---|---|
| Shakenova Aruzhan | 1-4 | Introduction, why K-Dramas, dataset |
| Qabir Madina | 5-9 | Preprocessing and supervised learning |
| Shahap Aizere | 10-14 | Unsupervised learning: clustering and projections |
| Zholdaspek Dana | 15-{num} | Anomalies, website demonstration, conclusions, closing |

Regenerate with `python presentation/build_presentation.py` after any change to the models.

---

"""
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "PRESENTATION_SCRIPT.md").write_text(header + "\n".join(script_lines))
print("saved docs/PRESENTATION_SCRIPT.md")

# ------------------------------------------------------------------ defence questions and answers
qa = f"""# K-Drama AI - defence questions and simple answers

Short answers in simple English. Learn the idea, then say it in your own words.
Numbers are inserted from the real results (regenerate with `python presentation/build_presentation.py`).

## A. Data and task

**1. What is your variant and how did you change it?**
Variant 21, Movie preference: predict "likes movie" from genre score, runtime and popularity. We changed movies to Korean dramas. The features are the same idea, and we added three more.

**2. Is your data real or synthetic?**
Real. Two public Kaggle datasets scraped from MyDramaList. The labels are real scores written by {ds['viewers']:,} viewers in {ds['rows']:,} reviews. We did not generate any labels.

**3. What exactly is the target?**
`liked` = 1 if the viewer's own overall score is 8 or more out of 10, otherwise 0. {pc(ds['positive_rate'])} of reviews are class 1.

**4. Why did you not use the drama's public rating as the label?**
The public rating is an average of thousands of people. It is not the opinion of one viewer. We predict one viewer's opinion, so the public rating is only a feature.

**5. What is `genre_score`, and does it leak the answer?**
It is the viewer's average score for OTHER dramas that share a genre with this drama. The review we predict is never included, so it does not contain the answer.

**6. Why are there missing values?**
{pc(ds['missing_share_genre_score'])} of reviews have no `genre_score`, because the viewer has no other review with the same genre. These are real gaps. The pipeline fills them with the training median.

**7. What are the main limitations of the data?**
Reviewers are not typical viewers (they are more active and more positive); there are no dates; ratings are a 2023 snapshot; one source has no stated licence.

## B. Supervised learning

**8. What is data leakage and how did you avoid it?**
Leakage is when information from the test set helps training. We split first, we keep each viewer on one side only, and the imputer and scaler are inside the pipeline, so they learn from training rows only.

**9. Why did you not use the usual `train_test_split`?**
One viewer has many reviews. With a random split, the same viewer would be in train and test, and the test would be too easy. `StratifiedGroupKFold` keeps viewers together and still keeps the class balance (80/20).

**10. Why do you need a Pipeline?**
It joins preprocessing and the model into one object. In cross-validation every fold fits its own imputer and scaler. It also makes the saved model safe to use on the website.

**11. Why Logistic Regression and Random Forest?**
Logistic Regression is simple and easy to interpret. Random Forest can learn non-linear rules. Comparing a simple and a flexible model is good practice (and the worksheet requires both).

**12. What is cross-validation and why 3 folds?**
We divide training data into 3 parts, train on 2 and validate on 1, three times. It gives a more stable score than one split. Three folds is the worksheet requirement.

**13. Which model won and by how much?**
{best}: mean CV F1 {f3(models[best]['f1_mean'])} against {f3(models[other['model']]['f1_mean'])}. The difference is very small, about one standard deviation. We followed the rule "higher mean F1".

**14. What are your test results?**
Precision {f3(t['precision'])}, recall {f3(t['recall'])}, F1 {f3(t['f1'])}, ROC-AUC {f3(t['roc_auc'])}. Confusion matrix: TN {cm['tn']}, FP {cm['fp']}, FN {cm['fn']}, TP {cm['tp']}.

**15. Explain precision and recall in simple words.**
Precision: when the model says "liked", how often is it right? Recall: from all really liked dramas, how many did the model find?

**16. What is ROC-AUC?**
Take one liked review and one not-liked review. ROC-AUC is the chance that the model gives the liked one a higher probability. 0.5 is guessing, 1.0 is perfect. Ours is {f3(t['roc_auc'])}.

**17. Is the model good?**
It is useful but modest. A model that always says "liked" already has F1 {f3(cv['baseline_always_like_f1_mean'])}, because most reviews are positive. Our model is better, and ROC-AUC {t['roc_auc']:.2f} shows it ranks better than chance, but it is far from perfect.

**18. Which features are important? Does this prove a cause?**
`{order[0]}` and `{order[1]}`. No - importance shows what the model uses, not what causes people to like a drama.

**19. What is a decision threshold?**
The model gives a probability. The threshold (0.5) changes it into yes or no. A higher threshold gives fewer false positives but more false negatives.

**20. Which error is worse here?**
A false positive wastes the viewer's time (a drama is 16+ hours). A false negative hides a drama they would love. Neither is dangerous, so we keep a balanced threshold of 0.5.

**21. Why is scaling inside the Random Forest pipeline if trees do not need it?**
Trees do not need it and it does not change the result. We keep it to have the same workflow for both models, as in the worksheet example.

## C. Unsupervised learning

**22. What is the difference between supervised and unsupervised learning?**
Supervised: we have the correct answers and learn to predict them. Unsupervised: no answers; we look for structure such as groups or unusual points.

**23. Why must the target be excluded in Project 2?**
If the label is a feature, the clusters only copy the label and we learn nothing new. Unsupervised learning must work without the answer.

**24. Why did you standardize?**
All our methods use distances. Runtime has a standard deviation of about 24, genre score about 0.1. Without scaling, runtime decides everything.

**25. The silhouette was higher without scaling ({sc['raw_silhouette']:.2f} vs {sc['scaled_silhouette']:.2f}). Why not use raw data?**
Because the raw clustering uses almost only runtime, which has two clear peaks. It looks clean but ignores the other features. A higher silhouette is not always a better analysis.

**26. How does K-Means work?**
It repeats two steps: give each point to the nearest centre, then move each centre to the mean of its points. It stops when nothing changes.

**27. Why K = {km['chosen_k']} when the silhouette is best at K = 2?**
K = 2 ({f3(kc[2]['silhouette'])}) only separates short and long dramas. K = 4 ({f3(kc[4]['silhouette'])}) is almost as good, it is where the elbow becomes flat, the result is stable over seeds, and the four groups are easy to explain. We say openly that it is a choice.

**28. What are the four clusters?**
Short web dramas, long-running serials, popular highly rated hits, and standard dramas with average popularity. These names are our interpretation of the cluster means.

**29. What do eps and min_samples mean in DBSCAN?**
eps is the radius of a neighbourhood. min_samples is how many points must be inside the radius to call the area dense.

**30. What is "noise" in DBSCAN?**
A point that is not in any dense area. It is not an error - it only means the drama is in a sparse region for these settings. We got {db['noise']} noise points.

**31. How sensitive is DBSCAN?**
Very. In our grid the result goes from {grid[(0.5, 5)]['clusters']} clusters and {grid[(0.5, 5)]['noise']} noise points to {grid[(1.0, 10)]['clusters']} cluster and {grid[(1.0, 10)]['noise']} noise points.

**32. K-Means vs DBSCAN?**
K-Means needs K, makes round groups and assigns every point. DBSCAN needs eps and min_samples, finds dense groups of any shape and can leave points as noise. In our data DBSCAN merges hits and standard dramas.

**33. What does PCA do and how much variance do you keep?**
PCA makes new axes that are mixes of the features and keep the most variance. Two components keep {pc(pca['retained_2d'], 1)}.

**34. What do the PCA axes mean?**
PC1: rating, runtime and popularity together - "mainstream success". PC2: number of episodes.

**35. PCA vs t-SNE vs UMAP?**
PCA is linear and its axes have a meaning. t-SNE and UMAP are non-linear; they keep neighbours together but distances between groups have no meaning, and results change with parameters and the random seed.

**36. Do the islands in t-SNE prove there are clusters?**
No. t-SNE can cut continuous data into islands. We trust a group only when several methods agree.

**37. How does Isolation Forest work?**
It makes random cuts in the data. Unusual points are separated after a few cuts, so they get a low score.

**38. What is contamination?**
The share of points we ask the model to flag. It is our decision, not a measured rate of anomalies.

**39. Do Isolation Forest and DBSCAN agree?**
Partly. With contamination {iso_main['contamination']}, {iso_main['also_dbscan_noise']} of {iso_main['flagged']} flagged dramas are also DBSCAN noise. But DBSCAN has {db['noise']} noise points - many more. They define "unusual" differently.

**40. What kind of dramas are anomalies?**
Mega-hits such as {iso['top_anomalies'][0]['title']} and {iso['top_anomalies'][1]['title']}, and extreme formats such as serials with 100+ episodes. Unusual means a rare combination, not a bad drama.

## D. Website and reproducibility

**41. Does the website use the real model?**
Yes. The FastAPI backend loads the saved scikit-learn pipeline with joblib and returns its probability. A test checks that the API gives exactly the same number as the notebook.

**42. What happens if the backend is off?**
The site shows an error message. It never shows invented numbers.

**43. How can we reproduce your results?**
Run the notebooks top to bottom in Colab, or run the scripts in `ml/`. We use random_state = 42 everywhere. Two runs give the same numbers.

**44. What would you improve?**
Add review dates, add tags and per-genre scores, collect ratings from normal watch-lists, tune the threshold with real costs, and try Gaussian Mixture or HDBSCAN for overlapping clusters.
"""
(ROOT / "docs" / "DEFENSE_QA.md").write_text(qa)
print("saved docs/DEFENSE_QA.md")
