# K-Drama AI - defence questions and simple answers

Short answers in simple English. Learn the idea, then say it in your own words.
Numbers are inserted from the real results (regenerate with `python presentation/build_presentation.py`).

## A. Data and task

**1. What is your variant and how did you change it?**
Variant 21, Movie preference: predict "likes movie" from genre score, runtime and popularity. We changed movies to Korean dramas. The features are the same idea, and we added three more.

**2. Is your data real or synthetic?**
Real. Two public Kaggle datasets scraped from MyDramaList. The labels are real scores written by 4,277 viewers in 9,840 reviews. We did not generate any labels.

**3. What exactly is the target?**
`liked` = 1 if the viewer's own overall score is 8 or more out of 10, otherwise 0. 58% of reviews are class 1.

**4. Why did you not use the drama's public rating as the label?**
The public rating is an average of thousands of people. It is not the opinion of one viewer. We predict one viewer's opinion, so the public rating is only a feature.

**5. What is `genre_score`, and does it leak the answer?**
It is the viewer's average score for OTHER dramas that share a genre with this drama. The review we predict is never included, so it does not contain the answer.

**6. Why are there missing values?**
33% of reviews have no `genre_score`, because the viewer has no other review with the same genre. These are real gaps. The pipeline fills them with the training median.

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
Logistic Regression: mean CV F1 0.756 against 0.754. The difference is very small, about one standard deviation. We followed the rule "higher mean F1".

**14. What are your test results?**
Precision 0.733, recall 0.804, F1 0.767, ROC-AUC 0.786. Confusion matrix: TN 496, FP 334, FN 223, TP 916.

**15. Explain precision and recall in simple words.**
Precision: when the model says "liked", how often is it right? Recall: from all really liked dramas, how many did the model find?

**16. What is ROC-AUC?**
Take one liked review and one not-liked review. ROC-AUC is the chance that the model gives the liked one a higher probability. 0.5 is guessing, 1.0 is perfect. Ours is 0.786.

**17. Is the model good?**
It is useful but modest. A model that always says "liked" already has F1 0.733, because most reviews are positive. Our model is better, and ROC-AUC 0.79 shows it ranks better than chance, but it is far from perfect.

**18. Which features are important? Does this prove a cause?**
`genre_score` and `public_rating`. No - importance shows what the model uses, not what causes people to like a drama.

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

**25. The silhouette was higher without scaling (0.66 vs 0.29). Why not use raw data?**
Because the raw clustering uses almost only runtime, which has two clear peaks. It looks clean but ignores the other features. A higher silhouette is not always a better analysis.

**26. How does K-Means work?**
It repeats two steps: give each point to the nearest centre, then move each centre to the mean of its points. It stops when nothing changes.

**27. Why K = 4 when the silhouette is best at K = 2?**
K = 2 (0.302) only separates short and long dramas. K = 4 (0.285) is almost as good, it is where the elbow becomes flat, the result is stable over seeds, and the four groups are easy to explain. We say openly that it is a choice.

**28. What are the four clusters?**
Short web dramas, long-running serials, popular highly rated hits, and standard dramas with average popularity. These names are our interpretation of the cluster means.

**29. What do eps and min_samples mean in DBSCAN?**
eps is the radius of a neighbourhood. min_samples is how many points must be inside the radius to call the area dense.

**30. What is "noise" in DBSCAN?**
A point that is not in any dense area. It is not an error - it only means the drama is in a sparse region for these settings. We got 205 noise points.

**31. How sensitive is DBSCAN?**
Very. In our grid the result goes from 15 clusters and 563 noise points to 1 cluster and 83 noise points.

**32. K-Means vs DBSCAN?**
K-Means needs K, makes round groups and assigns every point. DBSCAN needs eps and min_samples, finds dense groups of any shape and can leave points as noise. In our data DBSCAN merges hits and standard dramas.

**33. What does PCA do and how much variance do you keep?**
PCA makes new axes that are mixes of the features and keep the most variance. Two components keep 68.1%.

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
Partly. With contamination 0.05, 43 of 55 flagged dramas are also DBSCAN noise. But DBSCAN has 205 noise points - many more. They define "unusual" differently.

**40. What kind of dramas are anomalies?**
Mega-hits such as Crash Landing on You and Goblin, and extreme formats such as serials with 100+ episodes. Unusual means a rare combination, not a bad drama.

## D. Website and reproducibility

**41. Does the website use the real model?**
Yes. The FastAPI backend loads the saved scikit-learn pipeline with joblib and returns its probability. A test checks that the API gives exactly the same number as the notebook.

**42. What happens if the backend is off?**
The site shows an error message. It never shows invented numbers.

**43. How can we reproduce your results?**
Run the notebooks top to bottom in Colab, or run the scripts in `ml/`. We use random_state = 42 everywhere. Two runs give the same numbers.

**44. What would you improve?**
Add review dates, add tags and per-genre scores, collect ratings from normal watch-lists, tune the threshold with real costs, and try Gaussian Mixture or HDBSCAN for overlapping clusters.
