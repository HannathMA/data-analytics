"""
API Routes for Smart Meter Clustering.
Exposes endpoints for file upload, dataset generation, ACF feature extraction,
K-Means clustering, cluster inspection, and elbow-curve analysis.
"""

import io
import os
import sys
from pathlib import Path
from typing import Dict, Any, Union, Optional
import numpy as np
import pandas as pd
from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename

# Ensure root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    UPLOAD_DIR,
    DEFAULT_K,
    MAX_LAG,
    SERIES_HOURS,
    ACF_SIGNIFICANCE_THRESHOLD,
    EVAL_K_MIN,
    EVAL_K_MAX,
)
from backend.core.cleaner import WaterfallCleaner
from backend.core.feature_extraction import (
    extract_acf_pearson,
    extract_acf_details,
    batch_extract_acf,
)
from backend.core.clustering import (
    SmartMeterClusterer,
    compute_elbow_curve,
    compute_clustering_evaluation_curve,
    find_optimal_k_silhouette,
)
from backend.core.sample_generator import generate_london_sample_dataset

api_bp = Blueprint("api", __name__, url_prefix="/api")

# In-memory storage for active session state
STATE: Dict[str, Any] = {
    "raw_df": None,
    "cleaned_df": None,
    "audit_log": None,
    "acf_features": None,
    "acf_feature_names": None,
    "acf_df": None,
    "clusterer": None,
    "cluster_results": None,
    "elbow_results": None,
    "dataset_source": "none",  # 'sample', 'uploaded'
}


def _ensure_active_dataset():
    """Loads existing processed sample data if available and state is empty."""
    if STATE["cleaned_df"] is None:
        processed_file = DATA_PROCESSED_DIR / "sample_london_smartmeter_cleaned.csv"
        raw_file = DATA_RAW_DIR / "sample_london_smartmeter_raw.csv"

        if processed_file.exists():
            STATE["cleaned_df"] = pd.read_csv(processed_file, index_col=0)
            if raw_file.exists():
                STATE["raw_df"] = pd.read_csv(raw_file, index_col=0)
            else:
                STATE["raw_df"] = STATE["cleaned_df"].copy()
            STATE["dataset_source"] = "sample_disk"
        else:
            # Auto-generate if not yet generated
            raw_df, cleaned_df = generate_london_sample_dataset(n_households=250, save_files=True)
            STATE["raw_df"] = raw_df
            STATE["cleaned_df"] = cleaned_df
            STATE["dataset_source"] = "sample_generated"


@api_bp.route("/status", methods=["GET"])
def get_status():
    """Returns current state of data and trained model."""
    _ensure_active_dataset()
    has_cleaned = STATE["cleaned_df"] is not None
    has_model = STATE["cluster_results"] is not None

    return jsonify({
        "status": "ready" if has_cleaned else "no_data",
        "dataset_source": STATE["dataset_source"],
        "cleaned_meters_count": len(STATE["cleaned_df"]) if has_cleaned else 0,
        "has_model": has_model,
        "k_clusters": STATE["cluster_results"]["metrics"]["n_clusters"] if has_model else None,
        "significance_threshold_95ci": round(float(ACF_SIGNIFICANCE_THRESHOLD), 4),
        "audit_log": STATE["audit_log"],
    })


@api_bp.route("/generate_sample", methods=["POST"])
def generate_sample():
    """Generates synthetic London SmartMeter dataset with waterfall edge cases."""
    try:
        req = request.get_json(silent=True) or {}
        n_households = int(req.get("n_households", 250))

        raw_df, cleaned_df = generate_london_sample_dataset(
            n_households=n_households,
            include_two_weeks=True,
            random_seed=42,
            save_files=True,
        )

        cleaner = WaterfallCleaner(series_hours=SERIES_HOURS)
        _, audit_log = cleaner.clean_matrix(raw_df, has_two_weeks=True)

        STATE["raw_df"] = raw_df
        STATE["cleaned_df"] = cleaned_df
        STATE["audit_log"] = audit_log
        STATE["dataset_source"] = "sample"
        # Reset previous cluster results
        STATE["acf_features"] = None
        STATE["cluster_results"] = None
        STATE["elbow_results"] = None

        return jsonify({
            "message": "Sample London SmartMeter dataset generated successfully.",
            "total_raw": len(raw_df),
            "total_cleaned": len(cleaned_df),
            "audit_log": audit_log,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/upload", methods=["POST"])
def upload_file():
    """
    Accepts CSV file upload containing meter readings.
    Executes Waterfall cleaner (missing values, 0-variance, vacant >200% jump)
    and stores processed matrix.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request."}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        filename = secure_filename(file.filename)
        save_path = UPLOAD_DIR / filename
        file.save(save_path)

        # Parse CSV
        df = pd.read_csv(save_path)

        # Check if first column is meter identifier
        meter_id_col = None
        if not pd.api.types.is_numeric_dtype(df.dtypes.iloc[0]):
            meter_id_col = df.columns[0]

        has_two_weeks = (
            df.shape[1] >= (SERIES_HOURS * 2 + (1 if meter_id_col else 0))
        )

        cleaner = WaterfallCleaner(series_hours=SERIES_HOURS)
        cleaned_df, audit_log = cleaner.clean_matrix(
            df, meter_id_col=meter_id_col, has_two_weeks=has_two_weeks
        )

        # Save cleaned data if filesystem is writable
        try:
            processed_path = DATA_PROCESSED_DIR / f"cleaned_{filename}"
            cleaned_df.to_csv(processed_path, index_label="meter_id")
        except OSError:
            pass

        STATE["raw_df"] = df
        STATE["cleaned_df"] = cleaned_df
        STATE["audit_log"] = audit_log
        STATE["dataset_source"] = f"uploaded:{filename}"
        STATE["acf_features"] = None
        STATE["cluster_results"] = None
        STATE["elbow_results"] = None

        return jsonify({
            "message": "Dataset uploaded and cleaned successfully.",
            "filename": filename,
            "raw_count": audit_log["initial_count"],
            "cleaned_count": audit_log["final_count"],
            "audit_log": audit_log,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _do_clustering(k: Union[int, str] = "auto", max_lag: int = MAX_LAG) -> Dict[str, Any]:
    """Helper to execute ACF extraction and K-Means clustering with Silhouette Score Maximization."""
    _ensure_active_dataset()
    if STATE["cleaned_df"] is None or len(STATE["cleaned_df"]) == 0:
        raise ValueError("No cleaned data available. Please upload or generate sample data first.")

    cleaned_df = STATE["cleaned_df"]
    meter_ids = [str(x) for x in cleaned_df.index]

    # 1. Feature extraction
    features, feature_names, features_df = batch_extract_acf(
        cleaned_df,
        max_lag=max_lag,
        meter_ids=meter_ids,
        include_lag_zero=True,
        apply_threshold=True,
    )

    STATE["acf_features"] = features
    STATE["acf_feature_names"] = feature_names
    STATE["acf_df"] = features_df

    # 2. K-Means clustering (supports integer k or 'auto' for Silhouette Score Maximization)
    clusterer = SmartMeterClusterer(n_clusters=k)
    results = clusterer.fit(
        features=features,
        meter_ids=meter_ids,
        raw_series_df=cleaned_df,
    )

    STATE["clusterer"] = clusterer
    STATE["cluster_results"] = results
    return results


@api_bp.route("/run_clustering", methods=["POST"])
def run_clustering():
    """
    Runs the complete pipeline:
    1. Extracts 24-lag Pearson ACF with 95% CI thresholding (|r| > 1.96 / sqrt(N)).
    2. Calculates optimal k via Silhouette Score Maximization (or accepts user override).
    3. Fits K-Means on the 24-dim/25-dim ACF feature space.
    4. Computes inertia, silhouette score, Davies-Bouldin index, and cluster centroids.
    5. Identifies exemplar households and behavioral profiles.
    """
    try:
        req = request.get_json(silent=True) or {}
        k_param = req.get("k", "auto")
        if k_param != "auto":
            try:
                k_val = int(k_param)
            except (ValueError, TypeError):
                k_val = "auto"
        else:
            k_val = "auto"

        max_lag = int(req.get("max_lag", MAX_LAG))

        results = _do_clustering(k=k_val, max_lag=max_lag)

        return jsonify({
            "message": "Clustering completed via Silhouette Score Maximization." if k_val == "auto" else "Clustering completed successfully.",
            "metrics": results["metrics"],
            "pca_variance_ratio": results["pca_variance_ratio"],
            "cluster_profiles": results["cluster_profiles"],
            "feature_names": STATE["acf_feature_names"],
            "significance_threshold_95ci": round(float(ACF_SIGNIFICANCE_THRESHOLD), 4),
            "selection_method": "silhouette_score_maximization" if k_val == "auto" else "manual",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/get_clusters", methods=["GET"])
def get_clusters():
    """Returns the current cluster model results."""
    try:
        _ensure_active_dataset()
        if STATE["cluster_results"] is None:
            _do_clustering(k="auto")

        if STATE["cluster_results"] is None:
            return jsonify({"error": "Clustering has not been run yet."}), 400

        return jsonify({
            "metrics": STATE["cluster_results"]["metrics"],
            "pca_variance_ratio": STATE["cluster_results"]["pca_variance_ratio"],
            "cluster_profiles": STATE["cluster_results"]["cluster_profiles"],
            "assignments": STATE["cluster_results"]["assignments"],
            "significance_threshold_95ci": round(float(ACF_SIGNIFICANCE_THRESHOLD), 4),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/elbow", methods=["GET"])
@api_bp.route("/evaluation_curve", methods=["GET"])
def get_elbow():
    """Computes inertia, silhouette, and Davies-Bouldin index across k in [2, 15] to optimize k via Silhouette Score Maximization."""
    try:
        _ensure_active_dataset()
        if STATE["acf_features"] is None:
            # Extract features first
            cleaned_df = STATE["cleaned_df"]
            features, _, _ = batch_extract_acf(
                cleaned_df,
                max_lag=MAX_LAG,
                include_lag_zero=True,
                apply_threshold=True,
            )
            STATE["acf_features"] = features

        if STATE["elbow_results"] is None:
            STATE["elbow_results"] = compute_clustering_evaluation_curve(
                STATE["acf_features"], k_min=EVAL_K_MIN, k_max=EVAL_K_MAX
            )

        return jsonify(STATE["elbow_results"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/meter/<meter_id>", methods=["GET"])
def get_meter_details(meter_id: str):
    """Returns raw 168h series, 24-lag ACF with 95% CI thresholds, and cluster assignment for a specific meter."""
    try:
        _ensure_active_dataset()
        cleaned_df = STATE["cleaned_df"]
        if cleaned_df is None or meter_id not in cleaned_df.index:
            return jsonify({"error": f"Meter ID '{meter_id}' not found in active dataset."}), 404

        series = cleaned_df.loc[meter_id].values
        acf_details = extract_acf_details(series, max_lag=MAX_LAG)

        # Lookup cluster assignment if model trained
        assigned_cluster = None
        if STATE["clusterer"] and STATE["clusterer"].labels is not None:
            meter_ids = [str(x) for x in cleaned_df.index]
            if meter_id in meter_ids:
                idx = meter_ids.index(meter_id)
                assigned_cluster = int(STATE["clusterer"].labels[idx])

        return jsonify({
            "meter_id": meter_id,
            "assigned_cluster": assigned_cluster,
            "raw_series_168h": [round(float(v), 4) for v in series],
            "mean_kwh": round(float(np.mean(series)), 4),
            "max_kwh": round(float(np.max(series)), 4),
            "min_kwh": round(float(np.min(series)), 4),
            "acf": acf_details,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/export_clusters", methods=["GET"])
def export_clusters():
    """Generates a downloadable CSV with meter IDs, cluster labels, and 24-lag ACF features."""
    try:
        _ensure_active_dataset()
        if STATE["cluster_results"] is None or STATE["acf_df"] is None:
            return jsonify({"error": "No clustering results available to export."}), 400

        acf_df = STATE["acf_df"].copy()
        acf_df["cluster_label"] = STATE["clusterer"].labels

        output = io.StringIO()
        acf_df.to_csv(output, index_label="meter_id")
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="smart_meter_acf_clusters.csv",
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500
