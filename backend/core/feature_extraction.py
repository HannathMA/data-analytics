"""
ACF Feature Extraction Pipeline using Pearson Correlation and 95% Confidence Interval Thresholding.
Transforms raw 168-hour consumption series into temporal behavioral signatures across 24 hourly lags.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from backend.config import MAX_LAG, CONFIDENCE_Z


def extract_acf_pearson(
    series: Union[List[float], np.ndarray],
    max_lag: int = MAX_LAG,
    include_lag_zero: bool = True,
    apply_threshold: bool = True,
) -> np.ndarray:
    """
    Computes Autocorrelation features up to max_lag using Pearson correlation.
    Filters non-significant values below the 95% confidence threshold (|r| <= 1.96 / sqrt(N)).

    Parameters:
    -----------
    series : array-like
        1D array or list of hourly consumption readings (e.g. 168 hours).
    max_lag : int
        Maximum number of lags to compute (default 24).
    include_lag_zero : bool
        If True, returns vector of size max_lag + 1 starting with r_0 = 1.0.
        If False, returns vector of size max_lag (lags 1 to max_lag).
    apply_threshold : bool
        If True, sets correlations where |r| <= 1.96 / sqrt(N) to 0.0.

    Returns:
    --------
    acf_vector : np.ndarray
        Array of ACF coefficients.
    """
    series = np.array(series, dtype=float)
    n = len(series)

    if n <= max_lag:
        raise ValueError(
            f"Series length ({n}) must be strictly greater than max_lag ({max_lag})."
        )

    # 95% confidence threshold under null hypothesis: SE = 1 / sqrt(N)
    threshold = CONFIDENCE_Z / np.sqrt(n)

    acf_vector = np.zeros(max_lag + 1)
    acf_vector[0] = 1.0  # Lag 0 is always 1.0

    for k in range(1, max_lag + 1):
        x = series[:-k]  # Original slice
        y = series[k:]   # Lagged slice

        # Check for constant sub-slices to avoid zero-division
        std_x = np.std(x)
        std_y = np.std(y)
        if std_x <= 1e-9 or std_y <= 1e-9:
            acf_vector[k] = 0.0
            continue

        r, _ = pearsonr(x, y)

        # Retain statistically significant correlation; set noise to 0.0
        if apply_threshold:
            if abs(r) > threshold:
                acf_vector[k] = r
            else:
                acf_vector[k] = 0.0
        else:
            acf_vector[k] = r

    if not include_lag_zero:
        return acf_vector[1:]
    return acf_vector


def extract_acf_details(
    series: Union[List[float], np.ndarray],
    max_lag: int = MAX_LAG,
) -> Dict[str, any]:
    """
    Returns both raw (unthresholded) and thresholded ACF values along with the 95% CI bound.
    Useful for diagnostic visualization and plotting confidence corridors.
    """
    series = np.array(series, dtype=float)
    n = len(series)
    threshold = CONFIDENCE_Z / np.sqrt(n)

    raw_acf = extract_acf_pearson(
        series, max_lag=max_lag, include_lag_zero=True, apply_threshold=False
    )
    filtered_acf = extract_acf_pearson(
        series, max_lag=max_lag, include_lag_zero=True, apply_threshold=True
    )

    significant_lags = [
        int(k) for k in range(1, max_lag + 1) if filtered_acf[k] != 0.0
    ]

    return {
        "lags": list(range(max_lag + 1)),
        "raw_acf": raw_acf.tolist(),
        "filtered_acf": filtered_acf.tolist(),
        "threshold": float(threshold),
        "upper_bound": float(threshold),
        "lower_bound": float(-threshold),
        "significant_lags": significant_lags,
        "n_samples": n,
    }


def batch_extract_acf(
    data: Union[pd.DataFrame, np.ndarray],
    max_lag: int = MAX_LAG,
    meter_ids: Optional[List[str]] = None,
    include_lag_zero: bool = True,
    apply_threshold: bool = True,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Computes ACF features for a batch of smart meters.

    Parameters:
    -----------
    data : pd.DataFrame or np.ndarray
        Shape (N, 168).
    max_lag : int
        Number of lags.
    meter_ids : list, optional
        Identifiers for each meter.
    include_lag_zero : bool
        Whether to retain lag 0.
    apply_threshold : bool
        Whether to apply 95% CI threshold.

    Returns:
    --------
    features : np.ndarray
        2D array of shape (N, max_lag + 1) or (N, max_lag).
    feature_names : list of str
        ['lag_0', 'lag_1', ..., 'lag_24'] or ['lag_1', ...].
    features_df : pd.DataFrame
        DataFrame with meter IDs as index and feature_names as columns.
    """
    if isinstance(data, pd.DataFrame):
        if meter_ids is None:
            meter_ids = [str(x) for x in data.index]
        matrix = data.values
    else:
        matrix = np.asarray(data)
        if meter_ids is None:
            meter_ids = [f"meter_{i+1:04d}" for i in range(len(matrix))]

    features_list = [
        extract_acf_pearson(
            row,
            max_lag=max_lag,
            include_lag_zero=include_lag_zero,
            apply_threshold=apply_threshold,
        )
        for row in matrix
    ]

    features = np.array(features_list)

    if include_lag_zero:
        feature_names = [f"lag_{k}" for k in range(max_lag + 1)]
    else:
        feature_names = [f"lag_{k}" for k in range(1, max_lag + 1)]

    features_df = pd.DataFrame(features, index=meter_ids, columns=feature_names)
    return features, feature_names, features_df
