# DatoScope

DatoScope is a multipage Streamlit application for synthetic data generation, dataset upload, preprocessing, exploratory data analysis, supervised learning, clustering, and model comparison.

## Highlights

- Generate datasets inside the app for regression or clustering
- Upload a single dataset or separate train/test files
- Clean train and test data with missing-value handling, outlier removal, duplicate removal, and scaling
- Run EDA with summary statistics, distributions, correlations, scatter plots, and variance ranking
- Train regression models: Linear Regression, Ridge, Lasso
- Train classification models: Logistic Regression, Random Forest, KNN
- Run clustering models: K-Means, DBSCAN, Hierarchical Clustering
- Compare regression, classification, and clustering results in dedicated pages
- Download trained supervised models as `.pkl`

## Data Input Modes

The sidebar supports three workflows:

1. `Generate Dataset`
   Create synthetic regression or clustering datasets with controls for:
   - dataset type
   - sample count
   - noise
   - number of clusters / arms
   - number of features
   - random seed

2. `Upload Single File`
   Upload one file and let the app create an internal train/test split during supervised modeling.

3. `Upload Train/Test`
   Upload a train file and an optional test file.
   If the test file is present, the app uses it directly instead of creating a split.

## Pages

- `app.py`
  Data input, preprocessing, dataset metadata, raw/clean previews
- `pages/1_EDA.py`
  Exploratory data analysis
- `pages/2_Supervised_Modeling.py`
  Regression and classification workflows
- `pages/3_Clustering.py`
  Clustering workflows and visualizations
- `pages/4_Comparison.py`
  Model comparison dashboard

## Project Structure

```text
DatoScope/
├── app.py
├── pages/
│   ├── 1_EDA.py
│   ├── 2_Supervised_Modeling.py
│   ├── 3_Clustering.py
│   └── 4_Comparison.py
├── utils/
│   ├── app_state.py
│   ├── data_input.py
│   ├── generators.py
│   ├── modeling.py
│   ├── preprocessing.py
│   └── ui.py
├── scripts/
│   ├── 01_generate_data.py
│   ├── 02_clean_data.py
│   ├── 03_eda.py
│   └── 04_visualization.py
├── train.py
└── requirements.txt
```

## Setup

Use Python 3.11 for the most reliable dependency compatibility.

```bash
cd /Users/mehakgupta/Desktop/sem2/Data_Visual/project/DatoScope
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run The App

```bash
cd /Users/mehakgupta/Desktop/sem2/Data_Visual/project/DatoScope
/Users/mehakgupta/Desktop/sem2/Data_Visual/project/DatoScope/.venv/bin/python -m streamlit run app.py
```

## Run The Offline Pipeline

```bash
cd /Users/mehakgupta/Desktop/sem2/Data_Visual/project/DatoScope
/Users/mehakgupta/Desktop/sem2/Data_Visual/project/DatoScope/.venv/bin/python train.py
```

This pipeline runs:

1. synthetic dataset generation
2. cleaning and preprocessing
3. EDA reporting
4. static plot generation

## Supported File Types

| Extension | Notes |
|---|---|
| `.csv` | Standard CSV upload |
| `.xlsx` / `.xls` | Excel upload with simple header detection |
| `.zip` | Must contain one CSV |
| `.data` | Headerless comma- or whitespace-delimited files |

## Notes

- Classification is inferred from the selected target column but can also be chosen manually in the supervised modeling page.
- Model export currently supports supervised models.
- Clustering runs on the train dataset only when a separate test file is present.
