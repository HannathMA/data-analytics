"""
Configuration settings for Smart Meter Clustering pipeline and application server.
"""

import os
from pathlib import Path
import numpy as np

# Base paths
CONFIG_FILE_PATH = Path(__file__).resolve()
BACKEND_DIR = CONFIG_FILE_PATH.parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# In serverless environments (like Vercel / AWS Lambda), the root filesystem is read-only.
# We use /tmp for uploads while preserving bundled sample datasets in data/
if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    UPLOAD_DIR = Path("/tmp/uploads")
else:
    UPLOAD_DIR = DATA_RAW_DIR / "uploads"

# Ensure directories exist safely
try:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass

# Smart Meter Time Series Parameters
SERIES_HOURS = 168  # 1 week = 7 days * 24 hours
HOURLY_FREQUENCY = "1h"

# Data Cleaning Waterfall Thresholds
DROP_ZERO_VARIANCE = True
DROP_ZERO_MEAN_MEDIAN = True
MAX_WEEK_ON_WEEK_INCREASE = 2.0  # >200% week-on-week increase considered vacant/anomalous

# ACF Feature Extraction Parameters
MAX_LAG = 24  # 24 hourly lags to capture 24-hour diurnal cyclicity
CONFIDENCE_LEVEL = 0.95
CONFIDENCE_Z = 1.96
# Statistical significance threshold under null hypothesis: |r| > 1.96 / sqrt(N)
ACF_SIGNIFICANCE_THRESHOLD = CONFIDENCE_Z / np.sqrt(SERIES_HOURS)  # ~0.15121

# K-Means Clustering Parameters
DEFAULT_K = 12  # Benchmark k value
OPTIMAL_K_SELECTION_METHOD = "silhouette"  # Silhouette Score Maximization replaces heuristic elbow method
RANDOM_STATE = 12345
KMEANS_MAX_ITER = 300
KMEANS_N_INIT = 10
EVAL_K_MIN = 2
EVAL_K_MAX = 15
ELBOW_K_MIN = EVAL_K_MIN
ELBOW_K_MAX = EVAL_K_MAX

# Server Configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = int(os.environ.get("PORT", 5050))
DEBUG_MODE = True
MAX_CONTENT_LENGTH = 64 * 1024 * 1024  # 64 MB upload limit
