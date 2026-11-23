"""Project 2 - Unsupervised Learning: hidden patterns of K-Dramas.

Run:  python ml/analyze_unsupervised.py
Creates:
  artifacts/unsupervised_results.json   every number, table and 2-D coordinate used by the report,
                                        slides and the "Drama Insights" web page
  artifacts/figures/p2_*.png            figures for the report and slides

No target label is used anywhere in this file. One observation = one drama.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from kdrama_data import load_tables

warnings.filterwarnings("ignore", category=UserWarning)     # UMAP prints n_jobs notes when seeded
warnings.filterwarnings("ignore", category=FutureWarning)
import umap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIGURES = ARTIFACTS / "figures"
SEED = 42
FEATURES = ["genre_score", "runtime_min", "popularity_log", "episodes_log", "public_rating"]
K_RANGE = list(range(2, 9))
CHOSEN_K = 4
DBSCAN_EPS, DBSCAN_MIN_SAMPLES = 0.8, 10
EPS_GRID, MIN_SAMPLES_GRID = [0.5, 0.6, 0.7, 0.8, 1.0], [5, 10, 15]
CONTAMINATIONS = [0.02, 0.05, 0.10]
MAIN_CONTAMINATION = 0.05

PALETTE = ["#c97b8e", "#9d8cc9", "#e0b973", "#5fa8a3", "#d98f5c", "#6f86c9", "#7a1f3d", "#b8b8c4"]
NOISE_COLOR = "#3a3942"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def add_analysis_features(dramas: pd.DataFrame) -> pd.DataFrame:
    """Log-transform the two very skewed columns (row-by-row, nothing is learned from the data)."""
    dramas = dramas.copy()
    dramas["popularity_log"] = np.log10(dramas["popularity_rank"])
    dramas["episodes_log"] = np.log10(dramas["episodes"])
    return dramas


def dbscan_summary(labels) -> dict:
    return {"clusters": int(len(set(labels) - {-1})), "noise": int((labels == -1).sum())}


def describe_cluster(z_row: pd.Series) -> str:
    """Turn a cluster's standardized means into a neutral, data-derived description."""
    words = {"genre_score": ("lower genre score", "higher genre score"),
             "runtime_min": ("shorter episodes", "longer episodes"),
             "popularity_log": ("more popular", "less popular"),          # low rank number = popular
             "episodes_log": ("fewer episodes", "more episodes"),
             "public_rating": ("lower public rating", "higher public rating")}
    parts = [words[f][int(v > 0)] for f, v in z_row.sort_values(key=np.abs, ascending=False).items()
             if abs(v) >= 0.5]
    return ", ".join(parts) if parts else "close to the overall average"


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- DATA (no target loaded)
    dramas, _ = load_tables(ROOT / "data" / "raw")
    dramas = add_analysis_features(dramas)
    X = dramas[FEATURES]
    missing = X.isna().sum()

    # ---------------------------------------------------------------- PREPROCESSING
    X_imputed = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=FEATURES)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_imputed)

    # Scaling experiment: same K and seed, raw units versus standardized units.
    raw_labels = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=SEED).fit_predict(X_imputed)
    scaled_labels = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=SEED).fit_predict(Xs)
    scaling_experiment = {
        "k": CHOSEN_K,
        "raw_silhouette": float(silhouette_score(X_imputed, raw_labels)),
        "scaled_silhouette": float(silhouette_score(Xs, scaled_labels)),
        "ari_raw_vs_scaled": float(adjusted_rand_score(raw_labels, scaled_labels)),
        "raw_feature_std": X_imputed.std().round(3).to_dict(),
    }

    # Imputation experiment. The real table has no missing values, so we hide 5% of runtime_min
    # ON PURPOSE (clearly artificial) and check whether mean or median imputation changes the clusters.
    rng = np.random.default_rng(SEED)
    hidden = rng.choice(len(X), size=int(0.05 * len(X)), replace=False)
    X_holes = X_imputed.copy()
    X_holes.loc[hidden, "runtime_min"] = np.nan
    imputation_experiment = {"hidden_values": int(len(hidden)), "feature": "runtime_min", "strategies": {}}
    for strategy in ["mean", "median"]:
        filled = SimpleImputer(strategy=strategy).fit_transform(X_holes)
        filled_scaled = StandardScaler().fit_transform(filled)
        labels = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=SEED).fit_predict(filled_scaled)
        imputation_experiment["strategies"][strategy] = {
            "fill_value": float(np.nanmean(X_holes["runtime_min"]) if strategy == "mean"
                                else np.nanmedian(X_holes["runtime_min"])),
            "silhouette": float(silhouette_score(filled_scaled, labels)),
            "ari_vs_complete_data": float(adjusted_rand_score(scaled_labels, labels)),
        }

    # ---------------------------------------------------------------- K-MEANS
    k_curve = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED)
        labels = model.fit_predict(Xs)
        k_curve.append({"k": k, "inertia": float(model.inertia_),
                        "silhouette": float(silhouette_score(Xs, labels)),
                        "smallest_cluster": int(np.bincount(labels).min())})
    kmeans = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=SEED)
    km_labels = kmeans.fit_predict(Xs)

    stability = []
    for seed in range(5):
        m = KMeans(n_clusters=CHOSEN_K, n_init=10, random_state=seed)
        lab = m.fit_predict(Xs)
        stability.append({"seed": seed, "inertia": float(m.inertia_),
                          "silhouette": float(silhouette_score(Xs, lab)),
                          "ari_vs_seed_42": float(adjusted_rand_score(km_labels, lab))})

    profile_means = X_imputed.assign(cluster=km_labels).groupby("cluster").mean()
    original_units = dramas.assign(cluster=km_labels).groupby("cluster")[
        ["genre_score", "runtime_min", "popularity_rank", "episodes", "public_rating", "year"]]
    z_means = pd.DataFrame(Xs, columns=FEATURES).assign(cluster=km_labels).groupby("cluster").mean()
    cluster_profiles = []
    for c in range(CHOSEN_K):
        members = dramas[km_labels == c]
        genre_share = (members["genres"].str.split("|").explode().value_counts() / len(members)).head(4)
        cluster_profiles.append({
            "cluster": c, "size": int(len(members)), "share": float(len(members) / len(dramas)),
            "description": describe_cluster(z_means.loc[c]),
            "mean": {k: float(v) for k, v in original_units.mean().loc[c].round(2).items()},
            "median": {k: float(v) for k, v in original_units.median().loc[c].round(2).items()},
            "z_mean": {k: float(v) for k, v in z_means.loc[c].round(3).items()},
            "top_genres": [{"genre": g, "share": float(s)} for g, s in genre_share.items()],
            "examples": members.sort_values("popularity_rank")["title"].head(5).tolist(),
        })

    # ---------------------------------------------------------------- DBSCAN
    neighbours = NearestNeighbors(n_neighbors=DBSCAN_MIN_SAMPLES).fit(Xs)
    k_distance = np.sort(neighbours.kneighbors(Xs)[0][:, -1])
    dbscan_grid = []
    for eps in EPS_GRID:
        for ms in MIN_SAMPLES_GRID:
            lab = DBSCAN(eps=eps, min_samples=ms).fit_predict(Xs)
            row = {"eps": eps, "min_samples": ms, **dbscan_summary(lab)}
            core = lab != -1
            # Silhouette is only defined for >= 2 clusters; noise points are excluded from it.
            row["silhouette_without_noise"] = (float(silhouette_score(Xs[core], lab[core]))
                                               if row["clusters"] >= 2 else None)
            dbscan_grid.append(row)
    db_labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(Xs)
    db_noise = db_labels == -1
    db_sizes = {str(k): int(v) for k, v in pd.Series(db_labels).value_counts().sort_index().items()}
    db_profiles = (dramas.assign(cluster=db_labels).groupby("cluster")[
        ["genre_score", "runtime_min", "popularity_rank", "episodes", "public_rating"]].mean().round(2))
    crosstab = pd.crosstab(pd.Series(km_labels, name="kmeans"), pd.Series(db_labels, name="dbscan"))
    ari_km_db = float(adjusted_rand_score(km_labels[~db_noise], db_labels[~db_noise]))

    # ---------------------------------------------------------------- PCA
    pca_full = PCA().fit(Xs)
    Xp = PCA(n_components=2).fit_transform(Xs)
    loadings = pd.DataFrame(pca_full.components_[:2], columns=FEATURES, index=["PC1", "PC2"])

    # ---------------------------------------------------------------- t-SNE
    tsne_runs = {}
    for perplexity in [10, 30, 50]:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=SEED, init="pca",
                    learning_rate="auto")
        tsne_runs[perplexity] = tsne.fit_transform(Xs)
    Xt = tsne_runs[30]

    # ---------------------------------------------------------------- UMAP
    umap_runs = {}
    for n_neighbors in [5, 15, 50]:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=SEED)
        umap_runs[n_neighbors] = reducer.fit_transform(Xs)
    Xu = umap_runs[15]

    # ---------------------------------------------------------------- ISOLATION FOREST
    iso_flags, iso_summary = {}, []
    for c in CONTAMINATIONS:
        iso = IsolationForest(contamination=c, random_state=SEED).fit(Xs)
        iso_flags[c] = iso.predict(Xs) == -1
        both = int((iso_flags[c] & db_noise).sum())
        iso_summary.append({
            "contamination": c, "flagged": int(iso_flags[c].sum()),
            "also_dbscan_noise": both,
            "only_isolation_forest": int((iso_flags[c] & ~db_noise).sum()),
            "only_dbscan_noise": int((~iso_flags[c] & db_noise).sum()),
            "disagreement": int((iso_flags[c] != db_noise).sum()),
            "share_of_flags_that_are_noise": float(both / max(1, iso_flags[c].sum())),
        })
    iso_main = IsolationForest(contamination=MAIN_CONTAMINATION, random_state=SEED).fit(Xs)
    iso_score = iso_main.decision_function(Xs)                 # lower = more unusual
    top_anomalies = (dramas.assign(anomaly_score=iso_score, dbscan_noise=db_noise)
                     .sort_values("anomaly_score").head(10))
    top_anomalies = top_anomalies[["title", "year", "genres", "runtime_min", "episodes",
                                   "popularity_rank", "public_rating", "genre_score",
                                   "anomaly_score", "dbscan_noise"]].round(3)

    # ---------------------------------------------------------------- SAVE
    points = []
    for i, row in dramas.iterrows():
        points.append({
            "title": row["title"], "year": int(row["year"]), "genres": row["genres"],
            "runtime_min": float(row["runtime_min"]), "episodes": int(row["episodes"]),
            "popularity_rank": int(row["popularity_rank"]), "public_rating": float(row["public_rating"]),
            "genre_score": float(row["genre_score"]),
            "pca": [round(float(Xp[i, 0]), 3), round(float(Xp[i, 1]), 3)],
            "tsne": [round(float(Xt[i, 0]), 3), round(float(Xt[i, 1]), 3)],
            "umap": [round(float(Xu[i, 0]), 3), round(float(Xu[i, 1]), 3)],
            "kmeans": int(km_labels[i]), "dbscan": int(db_labels[i]),
            "anomaly_score": round(float(iso_score[i]), 4),
            "iso": {str(c): bool(iso_flags[c][i]) for c in CONTAMINATIONS},
        })

    results = {
        "dataset": {"observations": int(len(dramas)), "unit": "one K-Drama", "features": FEATURES,
                    "missing_counts": {f: int(missing[f]) for f in FEATURES},
                    "describe_original_units": dramas[["genre_score", "runtime_min", "popularity_rank",
                                                       "episodes", "public_rating"]].describe().round(3).to_dict(),
                    "skew_before_log": {"popularity_rank": float(dramas["popularity_rank"].skew()),
                                        "episodes": float(dramas["episodes"].skew())},
                    "skew_after_log": {"popularity_log": float(dramas["popularity_log"].skew()),
                                       "episodes_log": float(dramas["episodes_log"].skew())},
                    "correlation": X.corr().round(3).to_dict()},
        "preprocessing": {"scaling_experiment": scaling_experiment,
                          "imputation_experiment_artificial": imputation_experiment},
        "kmeans": {"k_curve": k_curve, "chosen_k": CHOSEN_K,
                   "chosen_silhouette": float(silhouette_score(Xs, km_labels)),
                   "chosen_inertia": float(kmeans.inertia_),
                   "seed_stability": stability, "profiles": cluster_profiles,
                   "profile_feature_means": profile_means.round(3).to_dict(orient="index")},
        "dbscan": {"eps": DBSCAN_EPS, "min_samples": DBSCAN_MIN_SAMPLES, **dbscan_summary(db_labels),
                   "cluster_sizes": db_sizes, "grid": dbscan_grid,
                   "k_distance_percentiles": {str(p): float(np.percentile(k_distance, p))
                                              for p in [50, 75, 90, 95, 99]},
                   "profiles_original_units": {str(k): v for k, v in db_profiles.to_dict(orient="index").items()},
                   "kmeans_vs_dbscan_crosstab": {str(k): {str(c): int(v) for c, v in row.items()}
                                                 for k, row in crosstab.to_dict(orient="index").items()},
                   "ari_with_kmeans_excluding_noise": ari_km_db},
        "pca": {"explained_variance_ratio": pca_full.explained_variance_ratio_.tolist(),
                "cumulative": np.cumsum(pca_full.explained_variance_ratio_).tolist(),
                "retained_2d": float(pca_full.explained_variance_ratio_[:2].sum()),
                "loadings": loadings.round(3).to_dict(orient="index")},
        "tsne": {"perplexity": 30, "sensitivity_perplexities": [10, 30, 50]},
        "umap": {"n_neighbors": 15, "min_dist": 0.1, "sensitivity_n_neighbors": [5, 15, 50]},
        "isolation_forest": {"main_contamination": MAIN_CONTAMINATION, "summary": iso_summary,
                             "top_anomalies": top_anomalies.to_dict(orient="records")},
        "points": points,
        "seed": SEED,
    }
    (ARTIFACTS / "unsupervised_results.json").write_text(json.dumps(results))

    make_figures(dramas, X, Xs, k_curve, km_labels, k_distance, dbscan_grid, db_labels, pca_full, Xp,
                 tsne_runs, umap_runs, iso_flags, z_means)

    print("observations:", len(dramas), "| missing:", missing.to_dict())
    print("scaling:", scaling_experiment)
    print("imputation:", imputation_experiment)
    print(pd.DataFrame(k_curve).round(3).to_string(index=False))
    print(pd.DataFrame(stability).round(3).to_string(index=False))
    for p in cluster_profiles:
        print(p["cluster"], p["size"], p["description"], p["mean"], [g["genre"] for g in p["top_genres"]])
    print(pd.DataFrame(dbscan_grid).round(3).to_string(index=False))
    print("DBSCAN main:", db_sizes, "ARI vs KMeans (no noise):", round(ari_km_db, 3))
    print(crosstab)
    print("PCA:", np.round(pca_full.explained_variance_ratio_, 3), "\n", loadings.round(2))
    print(pd.DataFrame(iso_summary).round(3).to_string(index=False))
    print(top_anomalies.to_string(index=False))


def scatter(ax, xy, labels, title, noise_label=None):
    for c in sorted(set(labels)):
        mask = labels == c
        if c == noise_label:
            ax.scatter(xy[mask, 0], xy[mask, 1], s=9, c=NOISE_COLOR, marker="x", label="noise", alpha=0.7)
        else:
            ax.scatter(xy[mask, 0], xy[mask, 1], s=9, c=PALETTE[c % len(PALETTE)], label=f"cluster {c}",
                       alpha=0.8)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def make_figures(dramas, X, Xs, k_curve, km_labels, k_distance, dbscan_grid, db_labels, pca_full, Xp,
                 tsne_runs, umap_runs, iso_flags, z_means) -> None:
    raw_cols = ["genre_score", "runtime_min", "popularity_rank", "episodes", "public_rating"]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax, col in zip(axes, raw_cols):
        ax.hist(dramas[col], bins=30, color=PALETTE[0])
        ax.set_title(col)
    fig.suptitle("Feature distributions in original units")
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_distributions.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].boxplot([Xs[:, i] for i in range(Xs.shape[1])], tick_labels=FEATURES, flierprops={"markersize": 3})
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("Standardized features (box plots)")
    corr = X.corr()
    im = axes[1].imshow(corr, cmap="RdPu", vmin=-1, vmax=1)
    axes[1].set_xticks(range(len(corr)), corr.columns, rotation=30, ha="right")
    axes[1].set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            axes[1].text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    axes[1].grid(False)
    axes[1].set_title("Correlation matrix")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_boxplot_correlation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    sc = ax.scatter(dramas["runtime_min"], dramas["popularity_rank"], c=dramas["public_rating"],
                    cmap="RdPu", s=10)
    ax.set_yscale("log")
    ax.set_xlabel("runtime_min")
    ax.set_ylabel("popularity_rank (log scale, 1 = most popular)")
    ax.set_title("Runtime vs popularity, coloured by public rating")
    fig.colorbar(sc, label="public_rating")
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_runtime_popularity.png")
    plt.close(fig)

    ks = [r["k"] for r in k_curve]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    axes[0].plot(ks, [r["inertia"] for r in k_curve], marker="o", color="#7a1f3d")
    axes[0].set_title("Elbow method: inertia")
    axes[0].set_xlabel("K")
    axes[1].plot(ks, [r["silhouette"] for r in k_curve], marker="o", color=PALETTE[1])
    axes[1].axvline(CHOSEN_K, color="gray", ls="--")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("K")
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_elbow_silhouette.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(z_means.values, cmap="RdPu", aspect="auto")
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=20)
    ax.set_yticks(range(len(z_means)), [f"cluster {c}" for c in z_means.index])
    for i in range(z_means.shape[0]):
        for j in range(z_means.shape[1]):
            ax.text(j, i, f"{z_means.values[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    ax.grid(False)
    ax.set_title("K-Means cluster profiles (mean z-score per feature)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_cluster_profiles.png")
    plt.close(fig)

    grid = pd.DataFrame(dbscan_grid)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    axes[0].plot(k_distance, color="#7a1f3d")
    axes[0].axhline(DBSCAN_EPS, color="gray", ls="--", label=f"eps = {DBSCAN_EPS}")
    axes[0].set_title(f"k-distance plot (k = {DBSCAN_MIN_SAMPLES})")
    axes[0].set_xlabel("dramas sorted by distance")
    axes[0].legend(frameon=False)
    for ms, color in zip(MIN_SAMPLES_GRID, PALETTE):
        part = grid[grid["min_samples"] == ms]
        axes[1].plot(part["eps"], part["clusters"], marker="o", color=color, label=f"min_samples={ms}")
        axes[2].plot(part["eps"], part["noise"], marker="o", color=color, label=f"min_samples={ms}")
    axes[1].set_title("DBSCAN: number of clusters")
    axes[2].set_title("DBSCAN: number of noise points")
    for ax in axes[1:]:
        ax.set_xlabel("eps")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_dbscan_sensitivity.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ratio = pca_full.explained_variance_ratio_
    axes[0].bar(range(1, len(ratio) + 1), ratio, color=PALETTE[0], label="per component")
    axes[0].plot(range(1, len(ratio) + 1), np.cumsum(ratio), marker="o", color="#7a1f3d", label="cumulative")
    axes[0].set_xlabel("principal component")
    axes[0].set_title("PCA explained variance")
    axes[0].legend(frameon=False)
    scatter(axes[1], Xp, km_labels, "PCA projection coloured by K-Means")
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_pca.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    scatter(axes[0], Xp, km_labels, "PCA (linear)")
    scatter(axes[1], tsne_runs[30], km_labels, "t-SNE (perplexity = 30)")
    scatter(axes[2], umap_runs[15], km_labels, "UMAP (n_neighbors = 15)")
    axes[0].legend(frameon=False, markerscale=2)
    fig.suptitle("Three 2-D views of the same 5 standardized features, coloured by K-Means cluster")
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_projection_comparison.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, p in zip(axes[0], [10, 30, 50]):
        scatter(ax, tsne_runs[p], km_labels, f"t-SNE perplexity = {p}")
    for ax, n in zip(axes[1], [5, 15, 50]):
        scatter(ax, umap_runs[n], km_labels, f"UMAP n_neighbors = {n}")
    fig.suptitle("Projection sensitivity: the picture changes with the settings")
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_projection_sensitivity.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    scatter(axes[0], Xp, km_labels, f"K-Means (K = {CHOSEN_K})")
    scatter(axes[1], Xp, db_labels, f"DBSCAN (eps = {DBSCAN_EPS}, min_samples = {DBSCAN_MIN_SAMPLES})",
            noise_label=-1)
    flagged = iso_flags[MAIN_CONTAMINATION]
    axes[2].scatter(Xp[~flagged, 0], Xp[~flagged, 1], s=9, c="#c9c7d1", label="normal")
    axes[2].scatter(Xp[flagged, 0], Xp[flagged, 1], s=18, c="#7a1f3d", label="flagged")
    axes[2].set_title(f"Isolation Forest (contamination = {MAIN_CONTAMINATION})")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].grid(False)
    for ax in axes:
        ax.legend(frameon=False, markerscale=2, fontsize=8)
    fig.suptitle("K-Means, DBSCAN and Isolation Forest in the same PCA view")
    fig.tight_layout()
    fig.savefig(FIGURES / "p2_methods_in_pca.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
