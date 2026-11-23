# Requirements checklists - Variant 21 (Movie preference -> K-Dramas)

Source of truth: the two official worksheets (`Practice_8_Final_Project_1.docx`, `Final_Project_P2.docx`).
"Where" points to the notebook step (the reports repeat the same content).

## Variant 21 as written in the worksheets

| Worksheet | Text |
|---|---|
| Project 1 | "Prediction (binary classification): likes movie. Features: genre score, runtime, popularity. Dataset status: no exact source established; select a suitable OpenML dataset or create an explicitly synthetic dataset. Confirm the target is actually present before using a public dataset." |
| Project 2 | "Exploratory question: identify groups, low-dimensional structure and unusual observations using genre score, runtime, popularity." (same dataset status) |

## Project 1 - Supervised Learning (Practice 8 worksheet)

| # | Worksheet requirement | Status | Where |
|---|---|---|---|
| 1 | Define the object, target and three or more plausible features | Done - object = one viewer review; target `liked`; 6 features | Notebook title table, Step 3 |
| 2 | Public dataset with the stated target, or clearly identified synthetic data; never claim simulated labels are real | Done - real labels from MyDramaList reviews; nothing simulated | Step 2 |
| 3 | If the source lacks the exact target, request approval to adapt the task | **Action for the team:** ask the instructor to approve "movie -> K-Drama" and the label definition | Step 0 note |
| 4 | Binary target coded 0/1; document the binary grouping | Done - score >= 8 -> 1, else 0 | Step 3 |
| 5 | Inspect data; introduce/document missing readings; describe class balance | Done - real missing values in `genre_score` documented (none added artificially, as in the worked example); class balance reported | Step 3 |
| 6 | Visualize at least three features and explain the graphs | Done - 4 figures, each explained | Step 4 |
| 7 | Split 80/20 before fitting imputers or scalers; stratify when possible | Done - stratified and grouped by viewer | Step 5 |
| 8 | Logistic Regression and Random Forest Pipelines | Done - imputer -> scaler -> classifier | Step 6 |
| 9 | Confusion matrix, precision, recall, F1, ROC-AUC | Done - out-of-fold for both models, test set for the selected one | Steps 7 and 9 |
| 10 | Same stratified 3-fold CV for both models; select by mean F1 | Done - identical folds | Steps 7-8 |
| 11 | Evaluate the selected model on the untouched test set | Done - once | Step 9 |
| 12 | Interpret feature importance (no causal claims) | Done - coefficients, impurity importance, permutation importance | Step 10 |
| 13 | Predict one new record; discuss a decision threshold and the cost of errors | Done - probability, class, threshold table, FP/FN costs | Step 11 |
| 14 | Limitations, risks and at least one realistic improvement | Done | Step 12 |
| 15 | Student questions (section 5 of the worksheet) | Answered | Step 12 |
| 16 | Deliverables: one executable Colab notebook + short report with plots, scores, source citation, conclusion | Done | `notebooks/Project1_Supervised_KDrama.ipynb`, `reports/Report1_Supervised_Learning.docx` |

Rubric coverage: provenance/target/features (15), EDA and data quality (10), leakage-safe split (15), LR + RF (20), CV and fair selection (15), test metrics and plots (15), conclusion/limitations/reproducibility (10).

## Project 2 - Unsupervised Learning (Final Project P2 worksheet)

| # | Worksheet requirement | Status | Where |
|---|---|---|---|
| 1 | Object, observation unit, research question, at least three numerical features | Done - one drama; 5 features | Notebook title table |
| 2 | No target label for model fitting | Done - the review table is deleted right after loading | Step 2 |
| 3 | At least 100 observations; document missing values, distributions, feature engineering | Done - 1,094 dramas; 0 missing; log transforms documented | Steps 2-3 |
| 4 | At least three exploratory figures; justify standardization | Done - 3 figures; scaling justified and tested | Steps 3-4 |
| 5 | Step 4.1: feature dictionary, scaling experiment (fixed K and seed), compare imputation strategies | Done - imputation test is clearly marked as artificial | Step 4.1 |
| 6 | K-Means for multiple K; elbow and silhouette curves; select K; interpret profiles | Done - K = 2...8, K = 4 justified | Steps 5-6 |
| 7 | Task 6.1: manual centroid, stability over seeds, profiles in original units | Done | Task 6.1 |
| 8 | DBSCAN with multiple eps and min_samples; clusters, noise, sensitivity | Done - 5 x 3 grid, k-distance plot | Steps 7 and 7.1 |
| 9 | Compare DBSCAN with K-Means | Done - cross-table and ARI | Step 7.1 |
| 10 | PCA with explained variance (and loadings) | Done | Step 8 |
| 11 | t-SNE and UMAP; compare with PCA | Done - comparison table and figure | Steps 9-10 |
| 12 | Step 10.1: projection sensitivity (perplexity, n_neighbors) | Done | Step 10.1 |
| 13 | Isolation Forest with at least two contamination values; compare with DBSCAN noise | Done - 0.02, 0.05, 0.10; overlap and disagreement | Steps 11 and 11.1 |
| 14 | Step 12: combined summary table | Done | Step 12 |
| 15 | Parameter sensitivity, reproducibility, methodological limitations | Done | Throughout + Conclusions |
| 16 | Evidence-based findings, cautions, improvement, source citation (conclusion template) | Done | Conclusions |
| 17 | Student questions (section 5) | Answered | Conclusions |
| 18 | Deliverables: executable Google Colab notebook + short report with tables, plots, interpretation | Done | `notebooks/Project2_Unsupervised_KDrama.ipynb`, `reports/Report2_Unsupervised_Learning.docx` |

Rubric coverage: provenance/question/features (10), inspection/plots/preprocessing (15), K-Means (20), DBSCAN (15), PCA/t-SNE/UMAP (20), Isolation Forest (10), conclusions/limitations/reproducibility (10).

## Differences between the original prompt and the worksheets (worksheets win)

| Topic | Prompt | What we did and why |
|---|---|---|
| Labels | Expected that synthetic viewer preferences might be needed | Not needed: real individual scores exist. The worksheet prefers a public dataset "with the stated target". |
| Split | "80/20 with stratification" (simple `train_test_split`) | Stratified **and grouped by viewer** (`StratifiedGroupKFold`). Still 80/20 and stratified as the worksheet asks, but without viewer leakage. |
| Missing values | "Handle missing values" | The worksheet says "introduce/document missing readings"; its worked example adds that missingness should reflect the real data. We document the real gaps and add none. |
| Approval | Not mentioned | The worksheet requires approval when a task is adapted. **Please confirm the K-Drama adaptation with the instructor.** |
| File name | `Practice_8_Final_Project_1(1).docx` | The attached file was `Practice_8_Final_Project_1.docx`; same document. |
