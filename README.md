# 🔬 DatoScope — Interactive ML Platform

A full Streamlit application for data cleaning, EDA, regression with regularization, and clustering.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Features

### 📁 Data Overview
- Upload CSV, Excel (.xls/.xlsx), ZIP-compressed CSV, or headerless .data files
- Instant shape, missing-value, and duplicate summary
- Downloadable cleaned dataset

### 🔧 Preprocessing (sidebar)
- Missing value strategies: mean / median / mode / drop
- Outlier removal: IQR or Z-Score
- Scalers: Standard / MinMax / Robust
- Duplicate removal toggle
- Optional label/target column (kept unscaled)

### 🔍 EDA
- Descriptive statistics + skewness + kurtosis
- Interactive feature distributions (histograms)
- Correlation heatmap (Pearson)
- Configurable scatter plot with trendline
- Feature variance ranking

### 📈 Regression
| Model | Regularization |
|---|---|
| Linear Regression | — |
| Ridge Regression | α ∈ {0.001 … 100} |
| Lasso Regression | α ∈ {0.001 … 100} |

**Metrics:** R², CV R², RMSE, MAE, MSE  
**Plots:** Actual vs Predicted · Residual distribution · Coefficient bar chart

### 🔵 Clustering
| Algorithm | Key Parameters |
|---|---|
| K-Means | k (2–10), elbow curve |
| DBSCAN | ε, min_samples |
| Hierarchical (Agglomerative) | n_clusters, linkage, dendrogram |

**Metrics:** Silhouette Score ↑ · Davies-Bouldin Index ↓ · Calinski-Harabasz ↑  
**Visuals:** PCA-reduced 2-D scatter · Cluster size bar · Dendrogram (Hierarchical) · Elbow curve (K-Means)

### 🏆 Model Comparison
- Side-by-side metric tables (auto-highlighted winners)
- Grouped bar charts for R² / error / silhouette
- Radar chart (normalized across all metrics)
- Automated text recommendation + overfitting detection
- One-click download of the full comparison report (.txt)

---

## Supported File Types
| Extension | Notes |
|---|---|
| `.csv` | Any delimiter; large files handled automatically |
| `.xlsx` / `.xls` | Multi-row header auto-detection |
| `.zip` | Must contain exactly one `.csv` inside |
| `.data` | Headerless; comma or whitespace delimited |

---

## Project Structure
```
datascope_app/
├── app.py              ← Main Streamlit application
└── requirements.txt    ← Python dependencies
```

> For the synthetic dataset scripts, see the `Project/` directory (train.py + scripts/).