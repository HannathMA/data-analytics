# Smart Meter Behavioral Clustering (VoltCluster)

> **Segmenting domestic electricity consumers using 24-lag Pearson Autocorrelation Function (ACF) feature engineering, 95% Confidence Interval noise suppression, and K-Means clustering.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/backend-Flask-black.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## ⚡ Problem Formulation & Core Concept

In conventional smart meter time-series clustering, algorithms like **K-Means** evaluate consumption vectors as independent spatial dimensions using Euclidean distance:

$$\|\mathbf{x}_i - \mathbf{x}_j\|_2 = \sqrt{\sum_{t=1}^{168} (x_{i,t} - x_{j,t})^2}$$

Because Euclidean distance ignores chronological order and temporal dynamics:
* **Phase shifts distort similarity**: Two households with identical evening routines shifted by just one hour appear far apart.
* **Volume confounds behavior**: High-volume cyclical users get grouped with high-volume flat users.
* **Overlapping clusters**: Directly clustering raw or normalized hourly readings leads to heavily blended, uninformative clusters.

### The ACF Solution
This project transforms raw consumption volume into **temporal behavioral signatures**:
1. **24-Lag Pearson Autocorrelation (ACF)**: Calculates autocorrelation across 24 hourly lags, capturing daily cycles, morning routines, and evening peak recurrence.
2. **95% Confidence Interval Cutoff**: Treats non-significant correlations below $|r| \le \frac{1.96}{\sqrt{N}}$ ($\approx 0.1512$ for $N=168$) as stochastic noise and replaces them with $0.0$.
3. **Dimensionality Reduction**: Compresses the feature space from **168 dimensions down to 25** (an **85.1% reduction**), lowering compute costs while yielding compact, well-separated clusters.
4. **Optimal $k=12$ Clustering**: Matches research benchmarks that identified $k=12$ as the local minimum of the **Davies-Bouldin Index (DBI)** and the elbow point of inertia.

---

## 📂 Project Structure

```text
smart-meter-clustering/
│
├── app.py                        # Root server entry point (run: python app.py)
├── requirements.txt              # Project dependencies (pip install -r requirements.txt)
│
├── data/
│   ├── raw/                      # Raw smart meter hourly CSV files (168h series)
│   └── processed/                # Cleaned data (missing values, 0-variance, & >200% jumps removed)
│
├── backend/
│   ├── app.py                    # Flask REST application server & UI host
│   ├── config.py                 # App configuration (cluster count, thresholds, paths)
│   ├── requirements.txt          # Backend dependencies
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── cleaner.py            # Data cleaning pipeline (waterfall filtering logic)
│   │   ├── feature_extraction.py # 24-lag ACF calculator and 95% CI thresholding
│   │   ├── clustering.py         # K-Means model pipeline and centroid extraction
│   │   └── sample_generator.py   # Synthetic London SmartMeter dataset generator
│   │
│   └── api/
│       ├── __init__.py
│       └── routes.py             # API endpoints (/upload, /run_clustering, /get_clusters, etc.)
│
├── frontend/
│   ├── templates/
│   │   └── index.html            # User interface layout and dashboard structure
│   │
│   └── static/
│       ├── css/
│       │   └── style.css         # UI layout, grid system, and metric card styling
│       └── js/
│           ├── app.js            # DOM handlers, form submissions, and API fetch calls
│           └── charts.js         # Chart.js configs (raw series, ACF lags, centroids)
│
├── notebooks/
│   └── exploration.ipynb         # Jupyter notebook for ACF validation and elbow-curve analysis
│
└── README.md                     # Setup instructions, architecture overview, and usage guide
```

---

## 🔬 Mathematical Methodology

### 1. 168-Hour Window Selection
A contiguous 7-day period (Monday 00:00 to Sunday 23:00) is sliced for each meter:
$$\mathbf{x} = [x_1, x_2, \dots, x_{168}] \in \mathbb{R}^{168}$$

### 2. Waterfall Data Cleaning Pipeline
The raw dataset undergoes a 3-stage audit filter:
1. **Missing Recordings Filter**: Drops meters with null or missing hourly observations.
2. **Zero-Variance / Flat Meter Filter**: Drops meters with $\text{mean} = 0$, $\text{median} = 0$, or $\text{variance} \le 10^{-6}$ (disconnected or dead meters).
3. **Vacant Residence Filter**: Compares week 1 vs. week 2; drops meters with a **$>200\%$ week-on-week increase** ($(\text{week}_2 - \text{week}_1)/\text{week}_1 > 2.0$), identifying homes transitioning out of vacancy.

### 3. Pearson Autocorrelation Feature Extraction
For lag $k=0$, $r_0 = 1.0$. For lag $k \in \{1, 2, \dots, 24\}$:
* Original slice: $X = [x_1, x_2, \dots, x_{168-k}]$
* Lagged slice: $Y = [x_{1+k}, x_{2+k}, \dots, x_{168}]$
* Pearson's correlation coefficient:
$$r_k = \frac{\sum_{i=1}^{168-k} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{168-k} (X_i - \bar{X})^2 \sum_{i=1}^{168-k} (Y_i - \bar{Y})^2}}$$

### 4. 95% Confidence Interval Thresholding
Under the null hypothesis ($H_0$: zero autocorrelation), the standard error is:
$$\text{SE} = \frac{1}{\sqrt{N}} = \frac{1}{\sqrt{168}} \approx 0.0771$$

At a 95% confidence level ($Z = 1.96$):
$$\text{Threshold} = \frac{1.96}{\sqrt{168}} \approx 0.1512$$

$$\text{Filtered } r_k = \begin{cases} r_k, & \text{if } |r_k| > 0.1512 \\ 0.0, & \text{if } |r_k| \le 0.1512 \end{cases}$$

---

## 🚀 Quickstart Guide

### 1. Prerequisites & Environment Setup
Clone the repository and ensure Python 3.10+ is installed:

```bash
cd smart-meter-clustering
pip install -r requirements.txt
```

### 2. Launch the Application Server
Run the Flask server:

```bash
python app.py
```

Open your browser at **`http://127.0.0.1:5050`** to access the interactive dashboard.

* **One-Click Demo**: Click **"Load London Sample"** to instantly generate 250 households with realistic consumer archetypes and waterfall edge cases.
* **Upload Custom CSV**: Upload your own raw smart meter file (either 168h or 336h).
* **Interactive Visualizations**: Inspect individual household 168-hour curves, 24-lag ACF plots with 95% CI bounds, 12 centroid signatures, and the elbow validation curve.

---

## 📊 Kaggle Dataset Instructions

To run on the open-access London dataset:
1. Go to Kaggle and search for: **"SmartMeter Energy Consumption Data in London Households"**
2. Download `halfhourly_dataset.zip` (containing readings in `KWH/hh` across 5,500+ households).
3. Resample the half-hourly intervals into 1-hour sums:
   ```python
   from backend.core.cleaner import WaterfallCleaner
   cleaner = WaterfallCleaner()
   hourly_df = cleaner.resample_half_hourly_to_hourly(raw_df, meter_col='LCLid', time_col='DateTime', kwh_col='KWH/hh')
   ```
4. Slice a full 168-hour week (Monday 00:00 to Sunday 23:00) and upload it through the web UI or save to `data/raw/`.

---

## 🌐 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web dashboard UI |
| `GET` | `/api/status` | Current dataset and active model status |
| `POST` | `/api/generate_sample` | Generates synthetic 168h London dataset with edge cases |
| `POST` | `/api/upload` | Uploads CSV, executes waterfall cleaning, stores matrix |
| `POST` | `/api/run_clustering` | Runs ACF extraction and K-Means (`k`, `max_lag`) |
| `GET` | `/api/get_clusters` | Retrieves centroids, cluster sizes, and PCA projections |
| `GET` | `/api/meter/<meter_id>` | Returns raw 168h series, 24-lag ACF, and assigned cluster |
| `GET` | `/api/elbow` | Returns Inertia, Silhouette, and DBI curves for $k \in [2, 15]$ |
| `GET` | `/api/export_clusters` | Downloads CSV with meter IDs, cluster labels, and ACF features |

---

## 📈 Notebook Exploration

To run the step-by-step mathematical validation and plotting walkthrough:

```bash
jupyter notebook notebooks/exploration.ipynb
```

The notebook guides you through:
1. Loading raw smart meter series
2. Auditing the Waterfall cleaning logic
3. Calculating 24-lag Pearson ACF and significance filtering
4. Evaluating the Elbow Curve, Silhouette scores, and Davies-Bouldin Index
5. 2D PCA projection and visualizing the 12 behavioral centroids

---

## 💡 Practical Applications for Utilities

* **Dynamic Demand Response (DR)**: Identifying households with recurring 24-hour peaks (e.g., clusters with $r_{24} > 0.5$) for targeted peak-shaving incentives.
* **Time-of-Use (ToU) Tariffs**: Formulating customized tariff tiers based on day vs. evening activity profiles.
* **Substation Forecasting**: Aggregating consumer clusters to build high-accuracy local transformer load prediction models.
* **Prosumer & Solar Integration**: Detecting midday baseload dips to forecast distributed rooftop solar capacity.
