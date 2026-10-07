"""
K-Means Clustering Pipeline and Behavioral Profiling for Smart Meter ACF Signatures.
Computes cluster centroids, evaluation metrics (Silhouette, Davies-Bouldin Index, Inertia),
behavioral profiling, and exemplar household selection.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from backend.config import DEFAULT_K, RANDOM_STATE, KMEANS_MAX_ITER, KMEANS_N_INIT


class SmartMeterClusterer:
    """
    Manages K-Means clustering lifecycle on 24-lag ACF feature vectors.
    """

    def __init__(
        self,
        n_clusters: Union[int, str] = DEFAULT_K,
        random_state: int = RANDOM_STATE,
        max_iter: int = KMEANS_MAX_ITER,
        n_init: int = KMEANS_N_INIT,
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.model: Optional[KMeans] = None
        self.centroids: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.metrics: Dict[str, float] = {}
        self.cluster_profiles: List[Dict[str, any]] = []
        self.pca_2d: Optional[np.ndarray] = None

    def fit(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        meter_ids: Optional[List[str]] = None,
        raw_series_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, any]:
        """
        Fits K-Means on the extracted ACF features.

        Parameters:
        -----------
        features : np.ndarray or pd.DataFrame
            Shape (N, D), typically D=25 (lag 0 to 24).
        meter_ids : list of str, optional
            Meter IDs corresponding to the rows.
        raw_series_df : pd.DataFrame, optional
            Raw 168-hour consumption DataFrame for exemplar profile visualization.

        Returns:
        --------
        results : dict
            Comprehensive summary containing metrics, centroids, cluster assignments,
            behavioral personas, and 2D PCA projections.
        """
        if isinstance(features, pd.DataFrame):
            if meter_ids is None:
                meter_ids = [str(x) for x in features.index]
            X = features.values
        else:
            X = np.asarray(features, dtype=float)
            if meter_ids is None:
                meter_ids = [f"meter_{i+1:04d}" for i in range(len(X))]

        n_samples, n_features = X.shape

        # Resolve k: if 'auto', calculate optimal k via Silhouette Score Maximization
        if self.n_clusters == "auto" or self.n_clusters is None:
            best_k, _ = find_optimal_k_silhouette(X, random_state=self.random_state)
            k = min(best_k, n_samples)
        else:
            k = min(int(self.n_clusters), n_samples)
        self.n_clusters = k

        # Fit K-Means
        self.model = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
        )
        self.labels = self.model.fit_predict(X)
        self.centroids = self.model.cluster_centers_

        # Calculate Validation Metrics
        inertia = float(self.model.inertia_)
        sil_score = None
        dbi_score = None
        ch_score = None

        if 1 < k < n_samples:
            try:
                sil_score = float(silhouette_score(X, self.labels))
                dbi_score = float(davies_bouldin_score(X, self.labels))
                ch_score = float(calinski_harabasz_score(X, self.labels))
            except Exception:
                pass

        self.metrics = {
            "n_clusters": k,
            "n_samples": n_samples,
            "n_features": n_features,
            "inertia": round(inertia, 4),
            "silhouette_score": round(sil_score, 4) if sil_score is not None else None,
            "davies_bouldin_index": round(dbi_score, 4) if dbi_score is not None else None,
            "calinski_harabasz_score": round(ch_score, 4) if ch_score is not None else None,
        }

        # 2D PCA projection for visual scatter
        pca = PCA(n_components=2, random_state=self.random_state)
        self.pca_2d = pca.fit_transform(X)
        pca_variance_ratio = [round(float(v), 4) for v in pca.explained_variance_ratio_]

        # Generate Behavioral Profiles for each cluster
        self.cluster_profiles = []
        assignments = []

        for cluster_id in range(k):
            cluster_mask = self.labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_size = int(np.sum(cluster_mask))
            cluster_pct = round((cluster_size / max(1, n_samples)) * 100, 2)

            centroid = self.centroids[cluster_id]

            # Find exemplar (meter closest to cluster centroid in ACF space)
            if cluster_size > 0:
                cluster_points = X[cluster_indices]
                distances = np.linalg.norm(cluster_points - centroid, axis=1)
                best_idx_in_cluster = np.argmin(distances)
                exemplar_global_idx = cluster_indices[best_idx_in_cluster]
                exemplar_meter_id = meter_ids[exemplar_global_idx]
                exemplar_dist = float(distances[best_idx_in_cluster])
            else:
                exemplar_meter_id = None
                exemplar_dist = 0.0

            # Exemplar raw 168h series
            exemplar_series = []
            if raw_series_df is not None and exemplar_meter_id in raw_series_df.index:
                exemplar_series = raw_series_df.loc[exemplar_meter_id].values.tolist()

            # Profile characterization based on centroid ACF properties
            persona, description = self._characterize_centroid(centroid)

            self.cluster_profiles.append({
                "cluster_id": cluster_id,
                "size": cluster_size,
                "percentage": cluster_pct,
                "persona": persona,
                "description": description,
                "centroid_acf": [round(float(v), 4) for v in centroid],
                "exemplar_meter_id": exemplar_meter_id,
                "exemplar_distance": round(exemplar_dist, 4),
                "exemplar_raw_series": [round(float(v), 4) for v in exemplar_series] if exemplar_series else [],
                "meter_ids": [meter_ids[i] for i in cluster_indices[:50]],  # preview top 50
            })

        # Meter-level records
        for i in range(n_samples):
            assignments.append({
                "meter_id": meter_ids[i],
                "cluster": int(self.labels[i]),
                "pca_x": round(float(self.pca_2d[i, 0]), 4),
                "pca_y": round(float(self.pca_2d[i, 1]), 4),
            })

        return {
            "metrics": self.metrics,
            "pca_variance_ratio": pca_variance_ratio,
            "cluster_profiles": self.cluster_profiles,
            "assignments": assignments,
        }

    def _characterize_centroid(self, centroid: np.ndarray) -> Tuple[str, str]:
        """
        Derives an intuitive behavioral persona from the 24-lag centroid profile.
        """
        # Exclude lag 0 (always 1.0) for evaluation
        lags = centroid[1:] if len(centroid) == 25 else centroid
        lag_24 = lags[23] if len(lags) >= 24 else 0.0
        lag_12 = lags[11] if len(lags) >= 12 else 0.0
        lag_1 = lags[0] if len(lags) >= 1 else 0.0
        mean_acf = np.mean(np.abs(lags))

        if lag_24 > 0.45:
            if lag_12 > 0.3:
                return (
                    "Dual-Peak Diurnal Routine",
                    "Strong 24-hour cycle combined with 12-hour sub-harmonics, typical of morning departure and evening return routine.",
                )
            return (
                "Pronounced Evening Peak Recurrence",
                "High autocorrelation at lag 24 (>0.45), indicating sharp, highly repetitive evening household activity every day.",
            )
        elif lag_24 > 0.25:
            return (
                "Moderate Daily Cyclicity",
                "Consistent diurnal rhythm with moderate evening recurrence; regular residential consumption rhythm.",
            )
        elif lag_12 > 0.25 and lag_24 < 0.2:
            return (
                "Semi-Diurnal Shift",
                "Strong 12-hour periodicity with weaker 24-hour consistency, representing split shifts or hybrid occupancy.",
            )
        elif lag_1 > 0.5 and mean_acf < 0.15:
            return (
                "High Autoregressive Persistence",
                "High short-lag autocorrelation (lag 1-3) decaying rapidly, characteristic of sustained loads like heating/cooling.",
            )
        elif mean_acf < 0.08:
            return (
                "Aperiodic Baseline / Flat",
                "Near-zero autocorrelation across all lags, typical of base-load appliances, refrigeration, or vacant apartments.",
            )
        else:
            return (
                "Irregular Fluctuating Demand",
                "Variable temporal correlations without rigid 24-hour periodicity, reflecting irregular lifestyle or flexible occupancy.",
            )


def find_optimal_k_silhouette(
    features: Union[np.ndarray, pd.DataFrame],
    k_min: int = 2,
    k_max: int = 15,
    random_state: int = RANDOM_STATE,
) -> Tuple[int, float]:
    """
    Calculates the optimal number of clusters k by maximizing the Silhouette Score.
    Replaces the subjective heuristic elbow method with an objective mathematical metric.
    """
    curve_data = compute_clustering_evaluation_curve(
        features=features, k_min=k_min, k_max=k_max, random_state=random_state
    )
    return int(curve_data["optimal_k_silhouette"]), float(curve_data["max_silhouette"])


def compute_clustering_evaluation_curve(
    features: Union[np.ndarray, pd.DataFrame],
    k_min: int = 2,
    k_max: int = 15,
    random_state: int = RANDOM_STATE,
) -> Dict[str, any]:
    """
    Evaluates K-Means clustering across a range of k values to determine the optimal k
    using Silhouette Score Maximization, alongside Davies-Bouldin Index (DBI) and Inertia.
    """
    if isinstance(features, pd.DataFrame):
        X = features.values
    else:
        X = np.asarray(features, dtype=float)

    n_samples = len(X)
    k_max_eval = min(k_max, n_samples - 1)

    k_values = list(range(k_min, k_max_eval + 1))
    inertias = []
    silhouettes = []
    dbis = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=5, max_iter=200)
        labels = km.fit_predict(X)
        inertias.append(round(float(km.inertia_), 4))

        sil = silhouette_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        silhouettes.append(round(float(sil), 4))
        dbis.append(round(float(dbi), 4))

    # Optimal k determined via Silhouette Score Maximization
    best_sil_idx = int(np.argmax(silhouettes))
    optimal_k_sil = k_values[best_sil_idx]
    max_sil_score = silhouettes[best_sil_idx]

    best_dbi_idx = int(np.argmin(dbis))
    optimal_k_dbi = k_values[best_dbi_idx]

    return {
        "k_range": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "davies_bouldin": dbis,
        "optimal_k": optimal_k_sil,  # Primary selection via Silhouette Score Maximization
        "optimal_k_silhouette": optimal_k_sil,
        "max_silhouette": max_sil_score,
        "optimal_k_dbi": optimal_k_dbi,
        "selection_method": "silhouette_score_maximization",
    }


# Backwards compatibility alias
compute_elbow_curve = compute_clustering_evaluation_curve
