# K-Drama AI - presentation speaking script

One combined presentation for both projects, 18 slides. The text is written in simple English (B1 level).
Read it a few times and then say it in your own words - do not read from the paper.
All numbers come from the real results (artifacts/*.json).

| Speaker | Slides | Part |
|---|---|---|
| Shakenova Aruzhan | 1-4 | Introduction, why K-Dramas, dataset |
| Qabir Madina | 5-9 | Preprocessing and supervised learning |
| Shahap Aizere | 10-14 | Unsupervised learning: clustering and projections |
| Zholdaspek Dana | 15-18 | Anomalies, website demonstration, conclusions, closing |

Regenerate with `python presentation/build_presentation.py` after any change to the models.

---

### Slide 1 - Title
**Speaker: Shakenova Aruzhan**

Good morning. We are a team of four students, and our project is called K-Drama AI. Our variant is number 21, Movie Preference. We adapted it to Korean dramas. We built two machine learning projects, one supervised and one unsupervised, and also a website that shows the results. Each of us will present one part.

### Slide 2 - One topic, three connected parts
**Speaker: Shakenova Aruzhan**

Our work has three parts. Project 1 is supervised learning: we predict if a viewer will like a drama. Project 2 is unsupervised learning: we look for groups and unusual dramas without any labels. The third part is a website. It is not required, but it shows our real models in a friendly way. All three parts use the same data and the same Python code, so the numbers are always the same.

### Slide 3 - Variant 21 — Movie preference, adapted to Korean dramas
**Speaker: Shakenova Aruzhan**

The worksheet says: predict if a viewer likes a movie, using genre score, runtime and popularity. It also says that no exact dataset exists for this variant. So we could choose our own topic. We chose Korean dramas because we watch them and understand them, and because we found real public data. We kept the three required features and added three more. We adapted the task, so this should be approved by the instructor.

### Slide 4 - Real data from MyDramaList — nothing is simulated
**Speaker: Shakenova Aruzhan**

We use two public datasets from Kaggle. Both were collected from the website MyDramaList. The first one has drama information and more than ten thousand reviews, and each review has the viewer's own score. The second one has genres and the public rating. We join them by title and year. After joining we have 9,840 reviews from 4,277 viewers, and 1,094 dramas. A viewer 'likes' a drama if their own score is 8 or more. About 58% of reviews are positive. Important: the public rating of a drama is not a personal opinion, so we use it only as a feature, not as the label.

### Slide 5 - Features and a leakage-safe design
**Speaker: Qabir Madina**

Now the preprocessing. The most important feature is the genre score. For each review, it is the average score that the same viewer gave to other dramas of the same genres. We never include the review that we want to predict, so the feature does not contain the answer. If the viewer has no other reviews, the value is missing. This is 33% of rows, and these are real missing values. We split the data 80 to 20 before any preprocessing. All reviews of one viewer stay on the same side, so test viewers are really new. The imputer and the scaler are inside a pipeline, so they learn only from training data. This prevents data leakage.

### Slide 6 - Two models, one fair comparison
**Speaker: Qabir Madina**

We trained two models. Logistic Regression is a simple linear model. It is easy to explain. Random Forest is many decision trees that vote. It can learn more complex rules. Both models are in the same pipeline. We compare them with the same three folds of cross-validation, and the score is F1. We choose the model with the higher mean F1 on training data. Only after that we open the test set, one time.

### Slide 7 - What the features look like
**Speaker: Qabir Madina**

Here are the six features for liked and not liked reviews. We can see three things. First, liked reviews have a higher genre score and a higher public rating. Second, runtime has two peaks: short web dramas and normal one-hour dramas. Third, the features have very different scales, so we need scaling for Logistic Regression.

### Slide 8 - Logistic Regression was selected and tested once
**Speaker: Qabir Madina**

These are the results. In cross-validation, Random Forest has a mean F1 of 0.754 and Logistic Regression has 0.756. They are very close. We selected Logistic Regression because its F1 is higher. On the test set, precision is 0.733, recall is 0.804, F1 is 0.767 and ROC-AUC is 0.786. There are 334 false positives and 223 false negatives. We must be honest: a model that always says 'liked' has F1 about 0.73. So our model is better, but not by much. ROC-AUC shows more clearly that it is better than guessing.

### Slide 9 - What matters, and what a mistake costs
**Speaker: Qabir Madina**

Which features are important? All three methods show the same two features on top: the genre score of the viewer and the public rating of the drama. Runtime and episodes are almost not important. But importance is not causation. It only shows what the model uses. We also predicted a new record. The model gave a probability of 0.79. It is above the threshold of 0.5, so the class is 'liked'. A false positive means we recommend a drama and the viewer does not like it: lost time. A false negative means we hide a drama that the viewer would love. If we move the threshold to 0.6, precision goes up to 0.75 but recall goes down to 0.67.

### Slide 10 - Same data, no labels
**Speaker: Shahap Aizere**

Now Project 2. Here one row is one drama, and we have 1,094 dramas. We do not use the 'liked' label at all. We use five numeric features: genre score, runtime, popularity, episodes and public rating. Popularity and episodes are very skewed, so we use a logarithm. Then we standardize all features. This is important because all our methods use distances. Without scaling, runtime would decide everything. We tested it: without scaling the silhouette looks higher, 0.66, but the clusters use only runtime. With scaling it is 0.29, and all features matter.

### Slide 11 - Choosing K: we selected K = 4
**Speaker: Shahap Aizere**

For K-Means we tried K from 2 to 8. On the left is the elbow plot. The curve is smooth, and it becomes flatter after K equals 4. On the right is the silhouette score. The best value is at K equals 2, 0.302. But two clusters only show short dramas and long dramas, and we already know this. There is a second small peak at K equals 4, 0.285. So we chose K equals 4. It is almost as good and gives more useful groups. The silhouette is about 0.3, which is low. This means the groups overlap. We also tried five random seeds, and the result was stable.

### Slide 12 - Four groups of K-Dramas
**Speaker: Shahap Aizere**

This table shows the four groups in original units. Cluster 0 has 125 dramas: more episodes, lower genre score, less popular, lower public rating. Cluster 1 has 337 dramas: shorter episodes, fewer episodes, lower genre score, lower public rating, less popular. Cluster 2 has 280 dramas: more popular, higher public rating, longer episodes, higher genre score. Cluster 3 has 352 dramas: longer episodes. So we can describe them as long serials, short web dramas, popular hits, and standard dramas. These names are our interpretation. The algorithm only gives numbers. Romance and Drama are the top genres in every cluster, so the groups are about format and popularity, not about genre.

### Slide 13 - Density clusters, noise and parameter sensitivity
**Speaker: Shahap Aizere**

DBSCAN finds dense areas and does not need the number of clusters. Points in empty areas are called noise. We tested fifteen combinations of the two parameters. With eps 0.8 and min samples 10, we get 4 clusters and 205 noise points. But DBSCAN is very sensitive. With a small eps we get 15 clusters and 563 noise points. With a large eps we get only 1 cluster. Compared with K-Means: DBSCAN also finds the web dramas and the long serials, but it puts hits and standard dramas into one big cluster, because there is no empty space between them. Noise does not mean an error. It only means the drama is in a sparse area.

### Slide 14 - Three ways to see five dimensions in 2-D
**Speaker: Shahap Aizere**

We cannot draw five dimensions, so we use three methods to make a 2D picture. PCA is linear. The first two components keep 68.1% of the variance. The first axis is about success: rating, runtime and popularity together. The second axis is the number of episodes. t-SNE and UMAP are non-linear. They keep neighbours together, so we see separate islands. But we must be careful. In t-SNE and UMAP the distance between islands and the size of islands have no meaning, and the picture changes when we change the parameters. So an island is a hint, not proof of a real cluster. All three pictures agree that web dramas and long serials are separate.

### Slide 15 - Which dramas are unusual?
**Speaker: Zholdaspek Dana**

Isolation Forest looks for unusual dramas. It cuts the data randomly, and unusual points are separated very fast. We tested three contamination values. With 5 percent, the model flags 55 dramas, and 43 of them are also DBSCAN noise. So the two methods mostly agree about the most unusual dramas. But DBSCAN has 205 noise points, many more. The two methods have different ideas of 'unusual'. Who is unusual? The first group is mega-hits like Crash Landing on You and Goblin. The second group is extreme formats, for example dramas with more than 100 episodes. So an anomaly here is a rare combination of values. It is not a bad drama and not an error.

### Slide 16 - K-Drama AI — a real model behind a real interface
**Speaker: Zholdaspek Dana**

This is our website. It has four pages: Home, AI Predictor, Drama Insights and About. The frontend is made with Next.js and React. The backend is made with Python and FastAPI. The backend loads the real model that we trained and saved with joblib. When the user presses the button, the website sends the values to Python, and Python returns the probability. We do not calculate anything in JavaScript and we do not use fake numbers. If the backend is off, the website shows an error message. The Insights page shows the real clustering results from Project 2. Now we will show a short live demo.

### Slide 17 - What we learned — and what we cannot claim
**Speaker: Zholdaspek Dana**

Our conclusions. In Project 1, the model predicts if a viewer likes a drama with F1 0.77 and ROC-AUC 0.79 on new viewers. The viewer's taste for the genre and the public rating are the most useful features. In Project 2, we found that the format of a drama is the main structure: web dramas, long serials, hits and standard dramas. Our limitations: reviewers are not typical viewers. Many viewers have no history. The data has no dates. The clusters overlap and depend on parameters. And importance is not causation. In the future we can add dates, more features like tags, and try other models.

### Slide 18 - Questions?
**Speaker: Zholdaspek Dana**

To finish: each of us worked on one main part, but all of us studied both projects and can answer questions about any step. Thank you for your attention. We are ready for your questions.
