"""
Reusable modeling helpers for DatoScope.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier


def export_model_bytes(model, *, features: list[str], target: str, task_type: str) -> bytes:
    payload = {
        "model": model,
        "features": features,
        "target": target,
        "task_type": task_type,
    }
    return pickle.dumps(payload)


def infer_supervised_task(df: pd.DataFrame, target_col: str) -> str:
    target = df[target_col]
    if not pd.api.types.is_numeric_dtype(target):
        return "Classification"
    unique = target.nunique(dropna=True)
    if unique <= 10 and unique / max(len(target), 1) < 0.1:
        return "Classification"
    return "Regression"


def _prepare_supervised_splits(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None,
    *,
    features: list[str],
    target_col: str,
    test_size: float,
    random_seed: int,
):
    X_full = df_train[features]
    y_full = df_train[target_col]
    valid_train = X_full.notna().all(axis=1) & y_full.notna()
    X_full = X_full.loc[valid_train]
    y_full = y_full.loc[valid_train]

    if df_test is not None:
        test_valid = df_test[features].notna().all(axis=1) & df_test[target_col].notna()
        X_train, y_train = X_full, y_full
        X_test = df_test.loc[test_valid, features]
        y_test = df_test.loc[test_valid, target_col]
        split_method = "Uploaded test file"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_full,
            y_full,
            test_size=test_size,
            random_state=random_seed,
            stratify=y_full if y_full.nunique(dropna=True) <= max(20, int(len(y_full) * 0.2)) and y_full.nunique(dropna=True) > 1 else None,
        )
        split_method = f"Auto split from train file ({int(test_size * 100)}% test)"
    return X_full, y_full, X_train, X_test, y_train, y_test, split_method


def run_regression_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None,
    *,
    features: list[str],
    target_col: str,
    test_size: float,
    cv_folds: int,
    random_seed: int,
    run_lr: bool,
    run_ridge: bool,
    ridge_alpha: float,
    run_lasso: bool,
    lasso_alpha: float,
) -> dict:
    X_full, y_full, X_train, X_test, y_train, y_test, split_method = _prepare_supervised_splits(
        df_train,
        df_test,
        features=features,
        target_col=target_col,
        test_size=test_size,
        random_seed=random_seed,
    )

    results = {}
    models_to_run = []
    if run_lr:
        models_to_run.append(("Linear Regression", LinearRegression()))
    if run_ridge:
        models_to_run.append((f"Ridge (α={ridge_alpha})", Ridge(alpha=ridge_alpha)))
    if run_lasso:
        models_to_run.append((f"Lasso (α={lasso_alpha})", Lasso(alpha=lasso_alpha, max_iter=10000)))

    for name, model in models_to_run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_r2 = cross_val_score(model, X_full, y_full, cv=min(cv_folds, len(X_full)), scoring="r2").mean()
        row = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "R²": round(r2_score(y_test, y_pred), 4),
            "MSE": round(mean_squared_error(y_test, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
            "CV R²": round(cv_r2, 4),
            "split_method": split_method,
            "features": features,
            "target": target_col,
        }
        if hasattr(model, "coef_"):
            row["coef"] = dict(zip(features, model.coef_))
        results[name] = row
    return results


def run_classification_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None,
    *,
    features: list[str],
    target_col: str,
    test_size: float,
    cv_folds: int,
    random_seed: int,
    run_logreg: bool,
    run_rf: bool,
    rf_estimators: int,
    run_knn: bool,
    knn_neighbors: int,
) -> dict:
    X_full, y_full, X_train, X_test, y_train, y_test, split_method = _prepare_supervised_splits(
        df_train,
        df_test,
        features=features,
        target_col=target_col,
        test_size=test_size,
        random_seed=random_seed,
    )

    results = {}
    models_to_run = []
    if run_logreg:
        models_to_run.append(("Logistic Regression", LogisticRegression(max_iter=3000)))
    if run_rf:
        models_to_run.append((f"Random Forest ({rf_estimators})", RandomForestClassifier(n_estimators=rf_estimators, random_state=random_seed)))
    if run_knn:
        models_to_run.append((f"KNN ({knn_neighbors})", KNeighborsClassifier(n_neighbors=knn_neighbors)))

    for name, model in models_to_run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cv_acc = cross_val_score(model, X_full, y_full, cv=min(cv_folds, len(X_full)), scoring="accuracy").mean()
        results[name] = {
            "model": model,
            "y_test": y_test,
            "y_pred": y_pred,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            "CV Accuracy": round(cv_acc, 4),
            "Confusion Matrix": confusion_matrix(y_test, y_pred),
            "Report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
            "split_method": split_method,
            "features": features,
            "target": target_col,
        }
    return results


def run_clustering_models(
    df: pd.DataFrame,
    *,
    features: list[str],
    run_km: bool,
    k_val: int,
    run_db: bool,
    db_eps: float,
    db_min: int,
    run_hc: bool,
    hc_k: int,
    hc_link: str,
) -> dict:
    X_c = df[features].dropna()
    clust_res = {}
    algo_list = []
    if run_km:
        algo_list.append(("K-Means", KMeans(n_clusters=k_val, random_state=42, n_init=10)))
    if run_db:
        algo_list.append(("DBSCAN", DBSCAN(eps=db_eps, min_samples=db_min)))
    if run_hc:
        algo_list.append(("Hierarchical", AgglomerativeClustering(n_clusters=hc_k, linkage=hc_link)))

    for name, algo in algo_list:
        labels = algo.fit_predict(X_c)
        unique = np.unique(labels[labels != -1])
        n_clust = len(unique)
        row = {"labels": labels, "n_clusters": n_clust}
        if n_clust >= 2 and len(unique) < len(X_c):
            mask = labels != -1
            row["Silhouette"] = round(silhouette_score(X_c[mask], labels[mask]), 4) if mask.sum() > 1 else None
            row["Davies-Bouldin"] = round(davies_bouldin_score(X_c[mask], labels[mask]), 4) if mask.sum() > 1 else None
            row["Calinski-Harabasz"] = round(calinski_harabasz_score(X_c[mask], labels[mask]), 4) if mask.sum() > 1 else None
        else:
            row["Silhouette"] = None
            row["Davies-Bouldin"] = None
            row["Calinski-Harabasz"] = None
        clust_res[name] = row

    return {"results": clust_res, "X": X_c.reset_index(drop=True), "features": features}


def projection_for_plot(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, str, str]:
    if len(features) > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(df[features].dropna())
        return coords, f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
    values = df[features].dropna().values
    return values, features[0], features[1]
