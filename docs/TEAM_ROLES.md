# Team roles - four students

Names are set in `frontend/lib/team.ts`, `reports/build_reports.py`, `presentation/build_presentation.py`
and `notebooks/build_notebooks.py`; regenerate after changing them.

Everyone has one main area **and** one area to cross-check, so that each student can explain both projects.

| Team member | Main responsibility | Files to know by heart | Cross-checks | Slides |
|---|---|---|---|---|
| **Shakenova Aruzhan** - Data & project lead | Dataset search and sources, joining the two tables, `genre_score`, the label definition, limitations of the data | `ml/kdrama_data.py`, notebook Steps 1-3 of both projects | Project 2 preprocessing (Shahap Aizere) | 1-4 |
| **Qabir Madina** - Supervised learning | Split, pipelines, Logistic Regression, Random Forest, cross-validation, metrics, threshold | `ml/train_supervised.py`, Project 1 notebook Steps 5-11, Report 1 | Isolation Forest (Zholdaspek Dana) | 5-9 |
| **Shahap Aizere** - Unsupervised learning | Scaling, K-Means, DBSCAN, PCA, t-SNE, UMAP, sensitivity studies | `ml/analyze_unsupervised.py`, Project 2 notebook Steps 4-10, Report 2 | Supervised metrics (Qabir Madina) | 10-14 |
| **Zholdaspek Dana** - Anomalies, website & delivery | Isolation Forest, FastAPI backend, Next.js website, reports and slides generation | `backend/app/main.py`, `frontend/`, Project 2 notebook Step 11 | Data pipeline (Shakenova Aruzhan) | 15-18 |

## Before the defence - everyone should be able to

1. Run both notebooks in Google Colab from top to bottom.
2. Explain why we split by viewer and why preprocessing lives inside a Pipeline.
3. Read a confusion matrix and say what precision, recall, F1 and ROC-AUC mean.
4. Explain why K = 4 was chosen although the silhouette is best at K = 2.
5. Explain why t-SNE/UMAP islands are not proof of clusters.
6. Answer every question in `docs/DEFENSE_QA.md` in their own words.

## Suggested rehearsal plan

| When | What |
|---|---|
| Day 1 | Everyone runs both notebooks; each student reads "their" code line by line |
| Day 2 | Pairs explain their part to the cross-checker; swap |
| Day 3 | Full presentation run with a timer; practise the website demo twice |
| Day 4 | Quiz each other with `docs/DEFENSE_QA.md` |
