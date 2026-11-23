"use client";

import { useCallback, useEffect, useState } from "react";

/* ------------------------------------------------------------------ types (mirror the FastAPI JSON) */

export type ModelInput = {
  name: string;
  label: string;
  unit: string;
  help: string;
  train_min: number;
  train_max: number;
  train_median: number;
};

export type TestMetrics = {
  threshold: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
  precision: number;
  recall: number;
  f1: number;
  roc_auc: number;
  accuracy: number;
};

export type ModelInfo = {
  model_name: string;
  threshold: number;
  inputs: ModelInput[];
  dataset: { rows: number; viewers: number; dramas: number; positive_rate: number; missing_share_genre_score: number };
  split: { train_rows: number; test_rows: number; test_positive_rate: number };
  cross_validation: {
    models: Record<string, { f1_mean: number; f1_std: number; roc_auc_mean: number }>;
    baseline_always_like_f1_mean: number;
  };
  test: TestMetrics;
  interpretation: {
    permutation_importance_test_auc_drop: Record<string, { mean: number; std: number }>;
    random_forest_importance: Record<string, number>;
    logistic_coefficients_standardized: Record<string, number>;
  };
  threshold_study: { threshold: number; precision: number; recall: number; f1: number; fp: number; fn: number }[];
  disclosure: string;
};

export type PredictionRequest = {
  genre_score: number | null;
  runtime_min: number;
  popularity_rank: number;
  public_rating: number;
  episodes: number;
  year: number;
};

export type Prediction = {
  probability_like: number;
  predicted_class: 0 | 1;
  predicted_label: string;
  threshold: number;
  model_name: string;
  confidence_note: string;
  feature_effects: {
    feature: string;
    label: string;
    value: number;
    typical_value: number;
    probability_with_typical_value: number;
    effect: number;
  }[];
  warnings: string[];
  disclosure: string;
};

export type Drama = {
  title: string;
  year: number;
  genres: string;
  runtime_min: number;
  episodes: number;
  popularity_rank: number;
  public_rating: number;
};

export type InsightPoint = Drama & {
  genre_score: number;
  pca: [number, number];
  tsne: [number, number];
  umap: [number, number];
  kmeans: number;
  dbscan: number;
  anomaly_score: number;
  iso: Record<string, boolean>;
};

export type ClusterProfile = {
  cluster: number;
  size: number;
  share: number;
  description: string;
  mean: Record<string, number>;
  median: Record<string, number>;
  z_mean: Record<string, number>;
  top_genres: { genre: string; share: number }[];
  examples: string[];
};

export type Insights = {
  dataset: { observations: number; features: string[] };
  preprocessing: {
    scaling_experiment: { raw_silhouette: number; scaled_silhouette: number; ari_raw_vs_scaled: number };
  };
  kmeans: {
    k_curve: { k: number; inertia: number; silhouette: number; smallest_cluster: number }[];
    chosen_k: number;
    chosen_silhouette: number;
    seed_stability: { seed: number; inertia: number; silhouette: number; ari_vs_seed_42: number }[];
    profiles: ClusterProfile[];
  };
  dbscan: {
    eps: number;
    min_samples: number;
    clusters: number;
    noise: number;
    cluster_sizes: Record<string, number>;
    grid: { eps: number; min_samples: number; clusters: number; noise: number; silhouette_without_noise: number | null }[];
    ari_with_kmeans_excluding_noise: number;
  };
  pca: {
    explained_variance_ratio: number[];
    cumulative: number[];
    retained_2d: number;
    loadings: Record<string, Record<string, number>>;
  };
  tsne: { perplexity: number };
  umap: { n_neighbors: number; min_dist: number };
  isolation_forest: {
    main_contamination: number;
    summary: {
      contamination: number;
      flagged: number;
      also_dbscan_noise: number;
      only_isolation_forest: number;
      only_dbscan_noise: number;
      share_of_flags_that_are_noise: number;
    }[];
    top_anomalies: (Drama & { anomaly_score: number; dbscan_noise: boolean; genre_score: number })[];
  };
  points: InsightPoint[];
};

export type Summary = {
  supervised: {
    reviews: number;
    viewers: number;
    dramas: number;
    positive_rate: number;
    missing_share_genre_score: number;
    train_rows: number;
    test_rows: number;
    selected_model: string;
    cv_f1: Record<string, number>;
    baseline_f1: number;
    test: TestMetrics;
  };
  unsupervised: {
    dramas: number;
    features: string[];
    chosen_k: number;
    silhouette: number;
    dbscan_clusters: number;
    dbscan_noise: number;
    pca_retained_2d: number;
    iso_contamination: number;
    iso_flagged: number;
    iso_also_noise: number;
  };
  featured: (Drama & { kmeans: number })[];
};

/* ------------------------------------------------------------------ fetch helpers */

export const OFFLINE_MESSAGE =
  "The Python model service is not reachable. No prediction or statistic is shown, because this site never invents results. Start the backend (uvicorn app.main:app --port 8000) and try again.";

export class ApiError extends Error {
  code?: string;
  constructor(message: string, code?: string) {
    super(message);
    this.code = code;
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(OFFLINE_MESSAGE);
  }
  const text = await response.text();
  let body: unknown = null;
  try {
    body = JSON.parse(text);
  } catch {
    /* not JSON: the proxy could not reach FastAPI */
  }
  if (!response.ok || body === null) {
    const detail = (body as { detail?: unknown } | null)?.detail;
    if (response.status === 422 && Array.isArray(detail)) {
      const first = detail[0] as { loc?: string[]; msg?: string };
      throw new ApiError(`Invalid input for "${first.loc?.slice(-1)[0]}": ${first.msg}.`);
    }
    if (typeof detail === "string") throw new ApiError(detail);
    if (detail && typeof detail === "object" && "message" in detail) {
      const d = detail as { message: string; code?: string };
      throw new ApiError(d.message, d.code);
    }
    throw new ApiError(OFFLINE_MESSAGE);
  }
  return body as T;
}

export function useApi<T>(path: string) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorCode, setErrorCode] = useState<string | undefined>();
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api<T>(path)
      .then(setData)
      .catch((e: ApiError) => { setData(null); setError(e.message); setErrorCode(e.code); })
      .finally(() => setLoading(false));
  }, [path]);

  useEffect(load, [load]);
  return { data, error, errorCode, loading, retry: load };
}

/* ------------------------------------------------------------------ formatting */

export const pct = (v: number, digits = 0) => `${(v * 100).toFixed(digits)}%`;
export const num = (v: number) => v.toLocaleString("en-US");
export const fixed = (v: number, digits = 3) => v.toFixed(digits);

/** Cluster colours shared by every chart and card. */
export const CLUSTER_COLORS = ["#e8a1ba", "#a79acb", "#e0b973", "#5fa8a3", "#d98f5c", "#6f86c9", "#b8b8c4"];
export const NOISE_COLOR = "#5b5564";
