"""
01_generate_data.py
DatoScope — Step 1: Synthetic Dataset Generation

Generates all datasets needed for clustering and regression experiments.
Datasets mirror the types shown in the HTML explorer:
  Clustering : Gaussian Blobs, Two Moons, Concentric Circles, Spiral,
               Anisotropic Blobs, Variable-Density Blobs
  Regression : Linear, Nonlinear (cubic), Sinusoidal,
               High-Dimensional (for Lasso/Ridge), Noisy variants

Run:
    python 01_generate_data.py

Output: datasets/*.csv  +  datasets/_registry.csv
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons, make_circles, make_regression

from utils import save_dataset, build_registry, DATASETS_DIR, ensure_dirs

# ── Global seed ──────────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CLUSTERING DATASETS
# ════════════════════════════════════════════════════════════════════════════

def gen_gaussian_blobs(n_samples=500, n_clusters=3, cluster_std=1.0) -> pd.DataFrame:
    """
    Isotropic Gaussian clusters — the 'easy mode' for K-Means.
    Each cluster is a spherical blob; well-separated when std is low.

    Parameters
    ----------
    n_samples   : total number of points
    n_clusters  : number of cluster centres
    cluster_std : spread of each cluster (higher = more overlap)
    """
    X, y = make_blobs(
        n_samples=n_samples,
        centers=n_clusters,
        cluster_std=cluster_std,
        random_state=SEED,
    )
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y

    # Inject configurable noise & outliers to mimic the HTML explorer
    _inject_missing(df, pct=0.05, feature_cols=["x1", "x2"])
    _inject_outliers(df, pct=0.03, feature_cols=["x1", "x2"])
    return df


def gen_moons(n_samples=500, noise=0.10) -> pd.DataFrame:
    """
    Two interleaving half-circles.
    K-Means fails here; DBSCAN handles it well.

    Parameters
    ----------
    noise : std of Gaussian noise added to the half-circles
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=SEED)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    _inject_missing(df, pct=0.05, feature_cols=["x1", "x2"])
    _inject_outliers(df, pct=0.03, feature_cols=["x1", "x2"])
    return df


def gen_circles(n_samples=500, noise=0.05) -> pd.DataFrame:
    """
    Two concentric rings (inner radius = 0.4 × outer).
    Centroid-based methods cannot separate these; good for DBSCAN demo.
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=SEED)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    _inject_missing(df, pct=0.05, feature_cols=["x1", "x2"])
    _inject_outliers(df, pct=0.03, feature_cols=["x1", "x2"])
    return df


def gen_spiral(n_samples=500, n_arms=3, n_turns=1.5, noise=0.15) -> pd.DataFrame:
    """
    Custom multi-arm spiral — matches the Spiral type in the HTML explorer.
    Very challenging for any distance-based clustering method.

    Parameters
    ----------
    n_arms  : number of spiral arms (= number of classes)
    n_turns : how many full rotations each arm makes
    noise   : Gaussian noise amplitude
    """
    pts_per_arm = n_samples // n_arms
    Xs, ys = [], []
    for arm in range(n_arms):
        theta  = np.linspace(0, n_turns * 2 * np.pi, pts_per_arm)
        r      = theta / (n_turns * 2 * np.pi)
        offset = (2 * np.pi / n_arms) * arm
        x1 = r * np.cos(theta + offset) + rng.normal(0, noise, pts_per_arm)
        x2 = r * np.sin(theta + offset) + rng.normal(0, noise, pts_per_arm)
        Xs.append(np.column_stack([x1, x2]))
        ys.extend([arm] * pts_per_arm)

    X = np.vstack(Xs)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = ys
    _inject_missing(df, pct=0.05, feature_cols=["x1", "x2"])
    _inject_outliers(df, pct=0.03, feature_cols=["x1", "x2"])
    return df


def gen_anisotropic_blobs(n_samples=500) -> pd.DataFrame:
    """
    Gaussian blobs passed through a linear transformation → tilted/elongated.
    Tests whether a clustering method handles non-spherical clusters.
    """
    X, y = make_blobs(n_samples=n_samples, centers=3, random_state=SEED)
    T    = np.array([[0.6, -0.6], [-0.4, 0.8]])   # stretch + rotate
    X    = X @ T
    df   = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    _inject_missing(df, pct=0.05, feature_cols=["x1", "x2"])
    _inject_outliers(df, pct=0.03, feature_cols=["x1", "x2"])
    return df


def gen_variable_density(n_samples=500) -> pd.DataFrame:
    """
    Three clusters with very different densities (tight, medium, loose).
    Motivates DBSCAN over K-Means — K-Means assumes equal-sized clusters.
    """
    # Split samples across clusters
    n1, n2 = n_samples // 4, n_samples // 3
    n3 = n_samples - n1 - n2

    c1 = rng.normal([0,  0],  0.3, (n1, 2))
    c2 = rng.normal([5,  5],  1.5, (n2, 2))
    c3 = rng.normal([-4, 4],  0.8, (n3, 2))

    X = np.vstack([c1, c2, c3])
    y = np.array([0]*n1 + [1]*n2 + [2]*n3)

    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["label"] = y
    _inject_missing(df, pct=0.05, feature_cols=["x1", "x2"])
    _inject_outliers(df, pct=0.03, feature_cols=["x1", "x2"])
    return df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — REGRESSION DATASETS
# ════════════════════════════════════════════════════════════════════════════

def gen_linear_regression(n_samples=500, noise=10.0) -> pd.DataFrame:
    """
    Simple linear relationship: y = w·x + ε
    Baseline for Linear Regression; Ridge and Lasso will match it exactly.
    """
    X, y = make_regression(n_samples=n_samples, n_features=1, noise=noise, random_state=SEED)
    df = pd.DataFrame({"x1": X[:, 0], "target": y})
    _inject_missing(df, pct=0.05, feature_cols=["x1"])
    _inject_outliers(df, pct=0.03, feature_cols=["x1"])
    return df


def gen_nonlinear_regression(n_samples=500, noise=5.0) -> pd.DataFrame:
    """
    Cubic polynomial: y = 2x³ – x² + 3x – 5 + ε
    Demonstrates where plain Linear Regression underfits and
    Ridge/Lasso with polynomial features help.
    """
    x  = np.linspace(-3, 3, n_samples)
    y  = 2*x**3 - x**2 + 3*x - 5
    y += rng.normal(0, noise, n_samples)
    df = pd.DataFrame({"x1": x, "target": y})
    _inject_missing(df, pct=0.05, feature_cols=["x1"])
    _inject_outliers(df, pct=0.03, feature_cols=["x1"])
    return df


def gen_sinusoidal_regression(n_samples=500, noise=0.30) -> pd.DataFrame:
    """
    y = sin(x) + ε  over [0, 4π]
    Highly nonlinear. Pure linear regression will drastically underfit.
    Matches the Nonlinear Regression type in the HTML explorer.
    """
    x  = np.linspace(0, 4 * np.pi, n_samples)
    y  = np.sin(x) + rng.normal(0, noise, n_samples)
    df = pd.DataFrame({"x1": x, "target": y})
    _inject_missing(df, pct=0.05, feature_cols=["x1"])
    _inject_outliers(df, pct=0.03, feature_cols=["x1"])
    return df


def gen_highdim_regression(n_samples=500, n_features=20, n_informative=5, noise=10.0):
    """
    Many features, but only n_informative are truly predictive.
    The rest are pure noise — ideal for showing Lasso's feature selection.

    Returns (features_df, coefficients_df)
    """
    X, y, coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        coef=True,
        random_state=SEED,
    )
    cols = [f"x{i+1}" for i in range(n_features)]
    df   = pd.DataFrame(X, columns=cols)
    df["target"] = y

    coef_df = pd.DataFrame({"feature": cols, "true_coefficient": coef})

    _inject_missing(df, pct=0.05, feature_cols=cols)
    _inject_outliers(df, pct=0.03, feature_cols=cols)
    return df, coef_df


def gen_noisy_regression(n_samples=500, noise_levels=(1.0, 5.0, 20.0)) -> dict:
    """
    Same linear signal (y = 3x + 7) at three noise levels.
    Lets you see how noise degrades model accuracy and stability.

    Returns dict {noise_level: DataFrame}
    """
    x      = np.linspace(-5, 5, n_samples)
    signal = 3 * x + 7
    dfs    = {}
    for sigma in noise_levels:
        y  = signal + rng.normal(0, sigma, n_samples)
        df = pd.DataFrame({"x1": x, "target": y})
        _inject_missing(df, pct=0.05, feature_cols=["x1"])
        dfs[sigma] = df
    return dfs


# ════════════════════════════════════════════════════════════════════════════
# HELPERS — noise / missing injection (mirrors HTML explorer sliders)
# ════════════════════════════════════════════════════════════════════════════

def _inject_missing(df: pd.DataFrame, pct: float, feature_cols: list):
    """Randomly set `pct` fraction of feature values to NaN."""
    for col in feature_cols:
        mask = rng.random(len(df)) < pct
        df.loc[mask, col] = np.nan


def _inject_outliers(df: pd.DataFrame, pct: float, feature_cols: list):
    """Replace `pct` fraction of rows with extreme values (±15σ)."""
    n_out = max(1, int(len(df) * pct))
    idx   = rng.choice(len(df), n_out, replace=False)
    for i in idx:
        col = feature_cols[rng.integers(0, len(feature_cols))]
        df.loc[i, col] = float(rng.normal(0, 1) * 15)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ensure_dirs()

    print("=" * 62)
    print("  DatoScope · 01_generate_data.py")
    print("=" * 62)

    # ── Clustering ──────────────────────────────────────────────────
    print("\n── Clustering datasets ─────────────────────────────────────")

    specs = [
        ("clustering_gaussian_blobs.csv",   gen_gaussian_blobs()),
        ("clustering_moons.csv",             gen_moons()),
        ("clustering_circles.csv",           gen_circles()),
        ("clustering_spiral.csv",            gen_spiral()),
        ("clustering_anisotropic.csv",       gen_anisotropic_blobs()),
        ("clustering_variable_density.csv",  gen_variable_density()),
    ]
    for fname, df in specs:
        path = save_dataset(df, fname)
        missing = df.isnull().sum().sum()
        print(f"  [OK] {fname:<42} shape={df.shape}  missing={missing}")

    # ── Regression ──────────────────────────────────────────────────
    print("\n── Regression datasets ──────────────────────────────────────")

    reg_specs = [
        ("regression_linear.csv",       gen_linear_regression()),
        ("regression_nonlinear.csv",     gen_nonlinear_regression()),
        ("regression_sinusoidal.csv",    gen_sinusoidal_regression()),
    ]
    for fname, df in reg_specs:
        path = save_dataset(df, fname)
        missing = df.isnull().sum().sum()
        print(f"  [OK] {fname:<42} shape={df.shape}  missing={missing}")

    # High-dimensional (returns tuple)
    df_hd, df_coef = gen_highdim_regression()
    save_dataset(df_hd,   "regression_highdim.csv")
    save_dataset(df_coef, "regression_highdim_true_coefs.csv")
    print(f"  [OK] {'regression_highdim.csv':<42} shape={df_hd.shape}  missing={df_hd.isnull().sum().sum()}")
    print(f"  [OK] {'regression_highdim_true_coefs.csv':<42} shape={df_coef.shape}")

    # Noisy variants
    noisy = gen_noisy_regression()
    for sigma, df in noisy.items():
        tag   = str(sigma).replace(".", "_")
        fname = f"regression_noisy_sigma{tag}.csv"
        save_dataset(df, fname)
        print(f"  [OK] {fname:<42} shape={df.shape}  σ={sigma}")

    # ── Registry ────────────────────────────────────────────────────
    print("\n── Building registry ────────────────────────────────────────")
    registry = build_registry("datasets")
    reg_path = os.path.join(DATASETS_DIR, "_registry.csv")
    registry.to_csv(reg_path, index=False)
    print(f"  [OK] _registry.csv  ({len(registry)} datasets)\n")

    print(registry[["filename", "task", "n_rows", "n_cols"]].to_string(index=False))
    print("\n" + "=" * 62)
    print(f"  All datasets saved to:  datasets/")
    print("=" * 62)


if __name__ == "__main__":
    main()
