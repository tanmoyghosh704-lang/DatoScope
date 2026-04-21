"""
DatoScope — Interactive ML & Visualization Platform
Streamlit application for data cleaning, EDA, regression, and clustering.
"""

import warnings
warnings.filterwarnings("ignore")

import io, os, zipfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DatoScope",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

  :root {
    --bg:       #0a0e1a;
    --surface:  #111827;
    --card:     #1a2235;
    --border:   #1e3a5f;
    --accent:   #00d4ff;
    --accent2:  #7c3aed;
    --green:    #10b981;
    --amber:    #f59e0b;
    --red:      #ef4444;
    --text:     #e2e8f0;
    --muted:    #64748b;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
  }

  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
  }

  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid var(--border);
  }

  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--muted);
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 13px;
    padding: 8px 18px;
    transition: all .2s;
  }

  .stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #000 !important;
    font-weight: 600;
  }

  div[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
  }

  div[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 12px; }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace;
    font-size: 1.4rem;
  }

  .stDataFrame, .stTable { border-radius: 10px; overflow: hidden; }

  .stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: #fff !important;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'DM Sans', sans-serif;
    padding: 10px 24px;
    transition: opacity .2s;
  }
  .stButton > button:hover { opacity: .85; }

  .stSelectbox > div > div,
  .stMultiSelect > div > div,
  .stSlider > div,
  .stNumberInput > div {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
  }

  h1,h2,h3 { font-family: 'Space Mono', monospace; }

  .ds-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
  }

  .ds-badge {
    display:inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .5px;
  }
  .ds-badge-cyan  { background: #00d4ff22; color: #00d4ff; border: 1px solid #00d4ff44; }
  .ds-badge-green { background: #10b98122; color: #10b981; border: 1px solid #10b98144; }
  .ds-badge-amber { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b44; }
  .ds-badge-red   { background: #ef444422; color: #ef4444; border: 1px solid #ef444444; }

  .winner-banner {
    background: linear-gradient(135deg, #10b98133, #00d4ff11);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
  }

  hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="DM Sans", color="#e2e8f0"),
    margin=dict(l=10, r=10, t=40, b=10),
)
PALETTE = px.colors.qualitative.Bold


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = dict(
        raw_df=None, clean_df=None, filename="",
        reg_results={}, cluster_results={},
        scaler_choice="Standard",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_uploaded_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    raw  = uploaded.read()

    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw), low_memory=False)

    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                st.error("No CSV found inside the ZIP archive.")
                st.stop()
            return pd.read_csv(io.BytesIO(zf.read(csvs[0])), low_memory=False)

    if name.endswith((".xls", ".xlsx")):
        engine = "xlrd" if name.endswith(".xls") else "openpyxl"
        df = pd.read_excel(io.BytesIO(raw), engine=engine)
        if all(df.dtypes == object):
            df2 = pd.read_excel(io.BytesIO(raw), engine=engine, header=1)
            if len(df2.select_dtypes("number").columns) > 0:
                df = df2
        return df

    if name.endswith(".data"):
        try:
            df = pd.read_csv(io.BytesIO(raw), header=None)
            if df.shape[1] == 1:
                raise ValueError
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), header=None, sep=r"\s+", engine="python")
        df.columns = [f"feat_{c}" for c in df.columns]
        return df

    st.error(f"Unsupported file type: {uploaded.name}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def clean_dataframe(
    df: pd.DataFrame,
    missing_strategy: str,
    outlier_method: str,
    scale_method: str,
    remove_dupes: bool,
    label_col: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    report = {"rows_in": len(df), "cols_in": len(df.columns), "actions": []}
    df = df.copy()

    # Drop columns that are entirely null
    all_null = df.columns[df.isnull().all()].tolist()
    if all_null:
        df.drop(columns=all_null, inplace=True)
        report["actions"].append(f"Dropped {len(all_null)} all-null column(s)")

    # Duplicates
    if remove_dupes:
        n = df.duplicated().sum()
        if n:
            df.drop_duplicates(inplace=True)
            report["actions"].append(f"Removed {n} duplicate row(s)")

    # Separate numeric / non-numeric
    num_cols  = df.select_dtypes(include="number").columns.tolist()
    if label_col and label_col in num_cols:
        num_cols.remove(label_col)
    cat_cols  = df.select_dtypes(exclude="number").columns.tolist()

    # Missing values
    if missing_strategy == "drop":
        before = len(df)
        df.dropna(subset=num_cols, inplace=True)
        report["actions"].append(f"Dropped {before - len(df)} row(s) with missing values")
    else:
        strat = {"mean": "mean", "median": "median", "mode": "most_frequent"}[missing_strategy]
        if num_cols:
            imp = SimpleImputer(strategy=strat if strat != "most_frequent" else "mean")
            df[num_cols] = imp.fit_transform(df[num_cols])
        if cat_cols:
            cat_imp = SimpleImputer(strategy="most_frequent")
            df[cat_cols] = cat_imp.fit_transform(df[cat_cols])
        report["actions"].append(f"Filled missing values with '{missing_strategy}'")

    # Outlier removal
    feat_cols = [c for c in num_cols if c != label_col]
    if outlier_method == "IQR" and feat_cols:
        before = len(df)
        mask = pd.Series([True] * len(df), index=df.index)
        for col in feat_cols:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            mask &= df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        df = df[mask]
        report["actions"].append(f"IQR: removed {before - len(df)} outlier row(s)")

    elif outlier_method == "Z-Score" and feat_cols:
        before = len(df)
        z = np.abs(stats.zscore(df[feat_cols].fillna(0)))
        df = df[(z < 3).all(axis=1)]
        report["actions"].append(f"Z-Score: removed {before - len(df)} outlier row(s)")

    # Scaling
    scalers = {"Standard": StandardScaler(), "MinMax": MinMaxScaler(), "Robust": RobustScaler()}
    scaler  = scalers.get(scale_method)
    if scaler and feat_cols:
        df[feat_cols] = scaler.fit_transform(df[feat_cols])
        report["actions"].append(f"Scaled with {scale_method}Scaler")

    report.update({"rows_out": len(df), "cols_out": len(df.columns)})
    return df, report


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div style='font-family:Space Mono,monospace;font-size:1.5rem;color:#00d4ff;font-weight:700;
                  letter-spacing:-1px;'>🔬 DatoScope</div>
      <div style='color:#64748b;font-size:12px;margin-top:4px;'>Interactive ML Platform</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Upload Dataset")
    uploaded = st.file_uploader(
        "CSV / Excel / ZIP / .data",
        type=["csv", "xls", "xlsx", "zip", "data"],
        label_visibility="collapsed",
    )

    if uploaded:
        df_raw = load_uploaded_file(uploaded)
        st.session_state.raw_df   = df_raw
        st.session_state.filename = uploaded.name
        st.session_state.clean_df = None   # reset on new upload
        st.session_state.reg_results     = {}
        st.session_state.cluster_results = {}
        st.success(f"✓ {uploaded.name}")
        st.caption(f"{df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

    st.divider()

    if st.session_state.raw_df is not None:
        st.markdown("#### Preprocessing")
        missing_strat = st.selectbox("Missing values", ["mean","median","mode","drop"])
        outlier_meth  = st.selectbox("Outlier method", ["IQR","Z-Score","None"])
        scale_meth    = st.selectbox("Scaler", ["Standard","MinMax","Robust"])
        remove_dupes  = st.checkbox("Remove duplicates", value=True)

        all_cols  = st.session_state.raw_df.columns.tolist()
        label_col = st.selectbox(
            "Label / target column (optional — keeps it unscaled)",
            ["— none —"] + all_cols,
        )
        label_col = None if label_col == "— none —" else label_col

        if st.button("⚙️  Clean & Preprocess", use_container_width=True):
            with st.spinner("Cleaning…"):
                clean, report = clean_dataframe(
                    st.session_state.raw_df,
                    missing_strat,
                    outlier_meth if outlier_meth != "None" else "none",
                    scale_meth,
                    remove_dupes,
                    label_col,
                )
                st.session_state.clean_df = clean
                st.session_state.clean_report = report
            st.success("Dataset cleaned!")

    st.divider()
    st.caption("DatoScope · MA25 Project")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_data, tab_eda, tab_reg, tab_clust, tab_compare = st.tabs([
    "📁  Data Overview",
    "🔍  EDA",
    "📈  Regression",
    "🔵  Clustering",
    "🏆  Model Comparison",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown("## Dataset Overview")

    if st.session_state.raw_df is None:
        st.info("👈 Upload a dataset using the sidebar to get started.")
        st.stop()

    df = st.session_state.raw_df
    cdf = st.session_state.clean_df

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",    f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Missing", f"{df.isnull().sum().sum():,}")
    c4.metric("Duplicates", f"{df.duplicated().sum():,}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Raw Data (first 200 rows)")
        st.dataframe(df.head(200), use_container_width=True, height=320)

    with col_b:
        st.markdown("#### Column Summary")
        summary = pd.DataFrame({
            "dtype":    df.dtypes.astype(str),
            "non-null": df.notnull().sum(),
            "missing%": (df.isnull().mean() * 100).round(2),
            "unique":   df.nunique(),
        })
        st.dataframe(summary, use_container_width=True, height=320)

    if cdf is not None:
        st.markdown("---")
        st.markdown("#### Cleaned Dataset Preview")
        report = st.session_state.get("clean_report", {})
        ca, cb, cc = st.columns(3)
        ca.metric("Rows after cleaning", f"{report.get('rows_out', len(cdf)):,}",
                  delta=f"{report.get('rows_out', len(cdf)) - report.get('rows_in', len(df)):,}")
        cb.metric("Cols after cleaning", f"{report.get('cols_out', len(cdf.columns))}")
        cc.metric("Actions taken", f"{len(report.get('actions',[]))}")

        if report.get("actions"):
            with st.expander("Cleaning log"):
                for a in report["actions"]:
                    st.write(f"• {a}")

        st.dataframe(cdf.head(200), use_container_width=True, height=280)

        csv_bytes = cdf.to_csv(index=False).encode()
        st.download_button("⬇ Download cleaned CSV", csv_bytes,
                           file_name="datascope_cleaned.csv", mime="text/csv")

    # Missing values heatmap
    st.markdown("---")
    st.markdown("#### Missing Values")
    miss = df.isnull().mean() * 100
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        st.success("No missing values in the raw dataset 🎉")
    else:
        fig = px.bar(
            x=miss.index, y=miss.values,
            labels={"x": "Column", "y": "Missing %"},
            color=miss.values, color_continuous_scale="plasma",
            **PLOTLY_THEME,
        )
        fig.update_layout(coloraxis_showscale=False, height=320)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("## Exploratory Data Analysis")

    df_eda = st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.raw_df
    if df_eda is None:
        st.info("Upload a dataset first.")
        st.stop()

    num_cols = df_eda.select_dtypes(include="number").columns.tolist()

    if not num_cols:
        st.warning("No numeric columns found.")
        st.stop()

    # ── Summary stats
    st.markdown("#### Summary Statistics")
    stats_df = df_eda[num_cols].describe().T
    stats_df["skewness"] = df_eda[num_cols].skew().round(3)
    stats_df["kurtosis"] = df_eda[num_cols].kurtosis().round(3)
    st.dataframe(stats_df.style.background_gradient(cmap="Blues", subset=["mean","std"]),
                 use_container_width=True)

    st.markdown("---")

    # ── Distributions
    st.markdown("#### Feature Distributions")
    sel_cols = st.multiselect("Choose features", num_cols, default=num_cols[:min(4, len(num_cols))])

    if sel_cols:
        n  = len(sel_cols)
        nc = min(3, n)
        nr = (n + nc - 1) // nc
        fig = make_subplots(rows=nr, cols=nc,
                            subplot_titles=sel_cols,
                            vertical_spacing=0.12)
        for i, col in enumerate(sel_cols):
            row, col_idx = divmod(i, nc)
            data = df_eda[col].dropna()
            fig.add_trace(go.Histogram(x=data, nbinsx=40, name=col,
                                       marker_color="#00d4ff", opacity=.75), row+1, col_idx+1)
        fig.update_layout(showlegend=False, height=300*nr, **PLOTLY_THEME)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Correlation heatmap
    st.markdown("#### Correlation Matrix")
    if len(num_cols) >= 2:
        corr  = df_eda[num_cols].corr()
        fig_c = px.imshow(
            corr, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            **PLOTLY_THEME,
        )
        fig_c.update_layout(height=500)
        st.plotly_chart(fig_c, use_container_width=True)

        # Top correlations
        mask = np.triu(np.ones(corr.shape, dtype=bool))
        corr_flat = corr.mask(mask).stack().reset_index()
        corr_flat.columns = ["Feature A", "Feature B", "Correlation"]
        corr_flat["abs"] = corr_flat["Correlation"].abs()
        top = corr_flat.nlargest(10, "abs").drop(columns="abs")
        with st.expander("Top 10 correlated pairs"):
            st.dataframe(top.reset_index(drop=True), use_container_width=True)

    st.markdown("---")

    # ── Scatter
    st.markdown("#### Scatter Plot")
    if len(num_cols) >= 2:
        sa, sb = st.columns(2)
        x_col = sa.selectbox("X axis", num_cols, index=0, key="eda_x")
        y_col = sb.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="eda_y")
        color_opts = ["— none —"] + df_eda.columns.tolist()
        color_col  = st.selectbox("Color by", color_opts, key="eda_color")
        fig_s = px.scatter(
            df_eda, x=x_col, y=y_col,
            color=None if color_col == "— none —" else color_col,
            opacity=0.65, trendline="ols",
            color_discrete_sequence=PALETTE,
            **PLOTLY_THEME,
        )
        fig_s.update_layout(height=420)
        st.plotly_chart(fig_s, use_container_width=True)

    # ── Variance ranking
    st.markdown("---")
    st.markdown("#### Feature Variance Ranking")
    var = df_eda[num_cols].var().sort_values(ascending=False)
    fig_v = px.bar(
        x=var.index, y=var.values,
        labels={"x": "Feature", "y": "Variance"},
        color=var.values, color_continuous_scale="viridis",
        **PLOTLY_THEME,
    )
    fig_v.update_layout(coloraxis_showscale=False, height=320)
    st.plotly_chart(fig_v, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tab_reg:
    st.markdown("## Regression with Regularization")

    df_r = st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.raw_df
    if df_r is None:
        st.info("Upload a dataset first.")
        st.stop()

    num_cols_r = df_r.select_dtypes(include="number").columns.tolist()
    if len(num_cols_r) < 2:
        st.warning("Need at least 2 numeric columns for regression.")
        st.stop()

    # Config
    ca, cb = st.columns([1, 2])
    with ca:
        st.markdown("#### Configuration")
        target_col = st.selectbox("🎯 Target (Y)", num_cols_r, index=len(num_cols_r)-1)
        feat_opts  = [c for c in num_cols_r if c != target_col]
        features   = st.multiselect("Feature columns (X)", feat_opts, default=feat_opts)
        test_size  = st.slider("Test split %", 10, 40, 20) / 100
        cv_folds   = st.slider("CV folds", 3, 10, 5)

        st.markdown("##### Linear Regression")
        run_lr = st.checkbox("Linear Regression", value=True)

        st.markdown("##### Ridge Regression")
        run_ridge = st.checkbox("Ridge", value=True)
        ridge_alpha = st.select_slider("Ridge α", [0.001,0.01,0.1,1,10,100], value=1.0) if run_ridge else 1.0

        st.markdown("##### Lasso Regression")
        run_lasso = st.checkbox("Lasso", value=True)
        lasso_alpha = st.select_slider("Lasso α", [0.001,0.01,0.1,1,10,100], value=0.1) if run_lasso else 0.1

        run_btn = st.button("🚀  Train Models", use_container_width=True)

    with cb:
        if not features:
            st.warning("Select at least one feature column.")
            st.stop()

        X = df_r[features].dropna()
        y = df_r.loc[X.index, target_col]

        if run_btn:
            results = {}
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            models_to_run = []
            if run_lr:    models_to_run.append(("Linear Regression",  LinearRegression()))
            if run_ridge: models_to_run.append((f"Ridge (α={ridge_alpha})", Ridge(alpha=ridge_alpha)))
            if run_lasso: models_to_run.append((f"Lasso (α={lasso_alpha})", Lasso(alpha=lasso_alpha, max_iter=10000)))

            for name, model in models_to_run:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_r2  = cross_val_score(model, X, y, cv=cv_folds, scoring="r2").mean()
                results[name] = {
                    "model":   model,
                    "y_test":  y_test,
                    "y_pred":  y_pred,
                    "R²":      round(r2_score(y_test, y_pred), 4),
                    "MSE":     round(mean_squared_error(y_test, y_pred), 4),
                    "RMSE":    round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
                    "MAE":     round(mean_absolute_error(y_test, y_pred), 4),
                    "CV R²":   round(cv_r2, 4),
                }
                if hasattr(model, "coef_"):
                    results[name]["coef"] = dict(zip(features, model.coef_))

            st.session_state.reg_results = results

        # Display results
        res = st.session_state.reg_results
        if not res:
            st.info("Configure & click **Train Models** to run regression.")
        else:
            # Metrics table
            st.markdown("#### Evaluation Metrics")
            metric_rows = []
            for name, r in res.items():
                metric_rows.append({"Model": name, "R²": r["R²"], "CV R²": r["CV R²"],
                                    "RMSE": r["RMSE"], "MAE": r["MAE"], "MSE": r["MSE"]})
            mdf = pd.DataFrame(metric_rows).set_index("Model")
            st.dataframe(
                mdf.style
                   .highlight_max(subset=["R²","CV R²"], color="#10b98133")
                   .highlight_min(subset=["RMSE","MAE","MSE"], color="#10b98133")
                   .format(precision=4),
                use_container_width=True,
            )

            # ── Per-model charts
            for name, r in res.items():
                st.markdown(f"---\n#### {name}")
                r1, r2, r3 = st.columns(3)
                r1.metric("R²",     r["R²"])
                r2.metric("RMSE",   r["RMSE"])
                r3.metric("CV R²",  r["CV R²"])

                ytest_arr  = np.array(r["y_test"])
                ypred_arr  = np.array(r["y_pred"])
                residuals  = ytest_arr - ypred_arr
                min_v, max_v = float(ytest_arr.min()), float(ytest_arr.max())

                fa, fb, fc = st.columns(3)

                # Actual vs Predicted
                with fa:
                    fig_ap = go.Figure()
                    fig_ap.add_trace(go.Scatter(
                        x=ytest_arr, y=ypred_arr, mode="markers",
                        marker=dict(color="#00d4ff", opacity=.6, size=5), name="Predictions",
                    ))
                    fig_ap.add_trace(go.Scatter(
                        x=[min_v, max_v], y=[min_v, max_v],
                        mode="lines", line=dict(color="#7c3aed", dash="dash"), name="Perfect fit",
                    ))
                    fig_ap.update_layout(title="Actual vs Predicted",
                                         xaxis_title="Actual", yaxis_title="Predicted",
                                         height=300, **PLOTLY_THEME)
                    st.plotly_chart(fig_ap, use_container_width=True)

                # Residual distribution
                with fb:
                    fig_res = go.Figure()
                    fig_res.add_trace(go.Histogram(x=residuals, nbinsx=30,
                                                   marker_color="#7c3aed", opacity=.8))
                    fig_res.update_layout(title="Residuals", xaxis_title="Residual",
                                          yaxis_title="Count", height=300, **PLOTLY_THEME)
                    st.plotly_chart(fig_res, use_container_width=True)

                # Coefficients
                with fc:
                    if "coef" in r:
                        coef_d = r["coef"]
                        coef_df = pd.DataFrame({"Feature": list(coef_d.keys()),
                                                "Coefficient": list(coef_d.values())})
                        coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=True).index)
                        fig_c = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                                       color="Coefficient", color_continuous_scale="RdBu",
                                       color_continuous_midpoint=0, **PLOTLY_THEME)
                        fig_c.update_layout(title="Coefficients", height=300, coloraxis_showscale=False)
                        st.plotly_chart(fig_c, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
with tab_clust:
    st.markdown("## Clustering Algorithms")

    df_c = st.session_state.clean_df if st.session_state.clean_df is not None else st.session_state.raw_df
    if df_c is None:
        st.info("Upload a dataset first.")
        st.stop()

    num_cols_c = df_c.select_dtypes(include="number").columns.tolist()
    if len(num_cols_c) < 2:
        st.warning("Need at least 2 numeric columns for clustering.")
        st.stop()

    # Config
    ca2, cb2 = st.columns([1, 2])
    with ca2:
        st.markdown("#### Configuration")
        clust_feats = st.multiselect("Feature columns", num_cols_c,
                                      default=num_cols_c[:min(5, len(num_cols_c))],
                                      key="clust_feats")
        color_by_col = st.selectbox("Color-by (ground truth)", ["— none —"] + df_c.columns.tolist())

        st.markdown("##### K-Means")
        run_km = st.checkbox("K-Means", value=True)
        k_val  = st.slider("K (clusters)", 2, 10, 3) if run_km else 3

        st.markdown("##### DBSCAN")
        run_db   = st.checkbox("DBSCAN", value=True)
        db_eps   = st.number_input("ε (eps)", min_value=0.01, max_value=10.0, value=0.5, step=0.05) if run_db else 0.5
        db_min   = st.slider("min_samples", 2, 20, 5) if run_db else 5

        st.markdown("##### Hierarchical (Agglomerative)")
        run_hc   = st.checkbox("Hierarchical", value=True)
        hc_k     = st.slider("n_clusters", 2, 10, 3, key="hc_k") if run_hc else 3
        hc_link  = st.selectbox("Linkage", ["ward","complete","average","single"]) if run_hc else "ward"

        clust_btn = st.button("🚀  Run Clustering", use_container_width=True)

    with cb2:
        if not clust_feats:
            st.warning("Select at least 2 feature columns.")
            st.stop()

        X_c = df_c[clust_feats].dropna()

        if clust_btn:
            clust_res = {}
            algo_list = []
            if run_km: algo_list.append(("K-Means",        KMeans(n_clusters=k_val, random_state=42, n_init=10)))
            if run_db: algo_list.append(("DBSCAN",          DBSCAN(eps=db_eps, min_samples=db_min)))
            if run_hc: algo_list.append(("Hierarchical",   AgglomerativeClustering(n_clusters=hc_k, linkage=hc_link)))

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
                    row["Silhouette"] = row["Davies-Bouldin"] = row["Calinski-Harabasz"] = None

                clust_res[name] = row

            st.session_state.cluster_results = {
                "results": clust_res, "X": X_c.reset_index(drop=True),
                "features": clust_feats,
            }

        cres = st.session_state.cluster_results
        if not cres:
            st.info("Configure & click **Run Clustering** to start.")
        else:
            results_c = cres["results"]
            X_plot    = cres["X"]
            feats_c   = cres["features"]

            # Metrics table
            st.markdown("#### Clustering Metrics")
            metric_rows_c = []
            for name, r in results_c.items():
                metric_rows_c.append({
                    "Algorithm":        name,
                    "Clusters found":   r["n_clusters"],
                    "Silhouette ↑":     r.get("Silhouette"),
                    "Davies-Bouldin ↓": r.get("Davies-Bouldin"),
                    "Calinski-Harabasz ↑": r.get("Calinski-Harabasz"),
                })
            cmdf = pd.DataFrame(metric_rows_c).set_index("Algorithm")
            st.dataframe(
                cmdf.style
                   .highlight_max(subset=["Silhouette ↑","Calinski-Harabasz ↑"], color="#10b98133")
                   .highlight_min(subset=["Davies-Bouldin ↓"], color="#10b98133")
                   .format(na_rep="N/A", precision=4),
                use_container_width=True,
            )

            # ── PCA for 2-D visualisation
            if X_plot.shape[1] > 2:
                pca = PCA(n_components=2)
                coords = pca.fit_transform(X_plot)
                xlab, ylab = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", \
                             f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
            else:
                coords = X_plot.values
                xlab, ylab = feats_c[0], feats_c[1]

            for name, r in results_c.items():
                st.markdown(f"---\n#### {name}  —  {r['n_clusters']} cluster(s)")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Clusters", r["n_clusters"])
                m2.metric("Silhouette", r.get("Silhouette") or "N/A")
                m3.metric("Davies-Bouldin", r.get("Davies-Bouldin") or "N/A")
                m4.metric("Calinski-Harabasz", r.get("Calinski-Harabasz") or "N/A")

                labels_arr = r["labels"]
                plot_df = pd.DataFrame({
                    "x": coords[:, 0], "y": coords[:, 1],
                    "Cluster": labels_arr.astype(str),
                })
                fig_cl = px.scatter(
                    plot_df, x="x", y="y", color="Cluster",
                    color_discrete_sequence=PALETTE,
                    labels={"x": xlab, "y": ylab},
                    opacity=0.75,
                    **PLOTLY_THEME,
                )
                fig_cl.update_traces(marker_size=6)
                fig_cl.update_layout(height=380,
                                     legend_title_text="Cluster",
                                     title=f"{name} — Cluster Assignments")
                st.plotly_chart(fig_cl, use_container_width=True)

                # Cluster sizes
                sizes = pd.Series(labels_arr).value_counts().sort_index()
                fig_sz = px.bar(
                    x=sizes.index.astype(str), y=sizes.values,
                    labels={"x": "Cluster", "y": "Count"},
                    color=sizes.values, color_continuous_scale="plasma",
                    **PLOTLY_THEME,
                )
                fig_sz.update_layout(title="Cluster sizes", coloraxis_showscale=False, height=260)
                st.plotly_chart(fig_sz, use_container_width=True)

                # Dendrogram for hierarchical
                if name == "Hierarchical":
                    st.markdown("##### Dendrogram (top 30 leaves)")
                    fig_dend, ax = plt.subplots(figsize=(14, 4))
                    fig_dend.patch.set_facecolor("#111827")
                    ax.set_facecolor("#1a2235")
                    Z = linkage(X_plot, method=hc_link)
                    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
                               color_threshold=0.6*max(Z[:,2]),
                               above_threshold_color="#64748b")
                    ax.tick_params(colors="#e2e8f0")
                    ax.spines[:].set_color("#1e3a5f")
                    plt.tight_layout()
                    st.pyplot(fig_dend, use_container_width=True)
                    plt.close()

                # Elbow curve for K-Means
                if name == "K-Means":
                    st.markdown("##### Elbow Curve (inertia)")
                    inertias = []
                    k_range  = range(2, min(11, len(X_plot)))
                    for ki in k_range:
                        km = KMeans(n_clusters=ki, random_state=42, n_init=10)
                        km.fit(X_plot)
                        inertias.append(km.inertia_)
                    fig_elbow = px.line(
                        x=list(k_range), y=inertias,
                        labels={"x": "k", "y": "Inertia"},
                        markers=True, **PLOTLY_THEME,
                    )
                    fig_elbow.update_traces(line_color="#00d4ff", marker_color="#7c3aed", marker_size=8)
                    fig_elbow.update_layout(title="Elbow Method", height=280)
                    st.plotly_chart(fig_elbow, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("## Model Comparison Dashboard")

    reg_res   = st.session_state.reg_results
    clust_res = st.session_state.cluster_results.get("results", {})

    if not reg_res and not clust_res:
        st.info("Train at least one regression or clustering model to see the comparison.")
        st.stop()

    # ── REGRESSION COMPARISON
    if reg_res:
        st.markdown("### 📈 Regression Models")
        rows = []
        for name, r in reg_res.items():
            rows.append({"Model": name, "R²": r["R²"], "CV R²": r["CV R²"],
                         "RMSE": r["RMSE"], "MAE": r["MAE"]})
        rdf = pd.DataFrame(rows)

        # Best model by R²
        best_r  = rdf.loc[rdf["R²"].idxmax()]
        st.markdown(f"""
        <div class="winner-banner">
          <div style="font-size:11px;color:#10b981;letter-spacing:1px;text-transform:uppercase;
                      font-family:'Space Mono',monospace;margin-bottom:4px;">Best Regression Model</div>
          <div style="font-size:1.4rem;font-weight:700;font-family:'Space Mono',monospace;color:#fff;">
            🥇 {best_r['Model']}
          </div>
          <div style="color:#94a3b8;font-size:13px;margin-top:6px;">
            R² = {best_r['R²']} · CV R² = {best_r['CV R²']} · RMSE = {best_r['RMSE']}
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("")

        fig_r = px.bar(
            rdf, x="Model", y=["R²", "CV R²"],
            barmode="group",
            color_discrete_sequence=["#00d4ff", "#7c3aed"],
            **PLOTLY_THEME,
        )
        fig_r.update_layout(title="R² Comparison", height=340, legend_title_text="Metric")
        st.plotly_chart(fig_r, use_container_width=True)

        fig_err = px.bar(
            rdf, x="Model", y=["RMSE", "MAE"],
            barmode="group",
            color_discrete_sequence=["#f59e0b", "#ef4444"],
            **PLOTLY_THEME,
        )
        fig_err.update_layout(title="Error Metrics (lower is better)", height=320, legend_title_text="Metric")
        st.plotly_chart(fig_err, use_container_width=True)

        # Radar chart
        st.markdown("#### Radar Chart — Normalized Metrics")
        radar_df = rdf.copy()
        for col in ["R²", "CV R²"]:
            max_v = radar_df[col].max()
            radar_df[col+"_norm"] = radar_df[col] / max_v if max_v else 0
        for col in ["RMSE", "MAE"]:
            max_v = radar_df[col].max()
            radar_df[col+"_norm"] = 1 - (radar_df[col] / max_v if max_v else 0)
        cats = ["R²", "CV R²", "RMSE (inv)", "MAE (inv)"]
        fig_rad = go.Figure()
        colors_ = ["#00d4ff", "#7c3aed", "#10b981"]
        for i, (_, row_) in enumerate(radar_df.iterrows()):
            vals = [row_["R²_norm"], row_["CV R²_norm"],
                    row_["RMSE_norm"], row_["MAE_norm"]]
            vals += [vals[0]]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]],
                fill="toself", name=row_["Model"],
                line_color=colors_[i % len(colors_)],
                fillcolor=colors_[i % len(colors_)] + "33",
            ))
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1],
                                       tickfont=dict(color="#64748b"),
                                       gridcolor="#1e3a5f"),
                       angularaxis=dict(tickfont=dict(color="#e2e8f0"),
                                        gridcolor="#1e3a5f"),
                       bgcolor="#111827"),
            height=420, paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", family="DM Sans"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_rad, use_container_width=True)

        # Recommendation
        st.markdown("#### 🔍 Recommendation")
        best = best_r["Model"]
        r2v  = best_r["R²"]
        rmse_v = best_r["RMSE"]
        if r2v > 0.85:
            quality = "excellent"
            badge_color = "green"
        elif r2v > 0.6:
            quality = "moderate"
            badge_color = "amber"
        else:
            quality = "poor"
            badge_color = "red"

        st.markdown(f"""
        <div class="ds-card">
          <b>{best}</b> achieves the highest R² of <code>{r2v}</code> with RMSE <code>{rmse_v}</code>.<br>
          Fit quality: <span class="ds-badge ds-badge-{badge_color}">{quality.upper()}</span><br><br>
          {'✅ The model generalises well — CV R² is close to test R².' if abs(r2v - best_r['CV R²']) < 0.05 else
           '⚠️ Gap between test and CV R² suggests some overfitting — consider stronger regularization.'}
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

    # ── CLUSTERING COMPARISON
    if clust_res:
        st.markdown("### 🔵 Clustering Models")
        crows = []
        for name, r in clust_res.items():
            crows.append({
                "Algorithm":          name,
                "Clusters":           r["n_clusters"],
                "Silhouette":         r.get("Silhouette"),
                "Davies-Bouldin":     r.get("Davies-Bouldin"),
                "Calinski-Harabasz":  r.get("Calinski-Harabasz"),
            })
        cdf2 = pd.DataFrame(crows)

        valid = cdf2.dropna(subset=["Silhouette"])
        if not valid.empty:
            best_c = valid.loc[valid["Silhouette"].idxmax()]
            st.markdown(f"""
            <div class="winner-banner">
              <div style="font-size:11px;color:#10b981;letter-spacing:1px;text-transform:uppercase;
                          font-family:'Space Mono',monospace;margin-bottom:4px;">Best Clustering Algorithm</div>
              <div style="font-size:1.4rem;font-weight:700;font-family:'Space Mono',monospace;color:#fff;">
                🥇 {best_c['Algorithm']}
              </div>
              <div style="color:#94a3b8;font-size:13px;margin-top:6px;">
                Silhouette = {best_c['Silhouette']} · Davies-Bouldin = {best_c['Davies-Bouldin']}
              </div>
            </div>""", unsafe_allow_html=True)
            st.markdown("")

        # Bar charts
        if not valid.empty:
            fig_sil = px.bar(
                cdf2.dropna(subset=["Silhouette"]),
                x="Algorithm", y="Silhouette",
                color="Algorithm",
                color_discrete_sequence=PALETTE,
                **PLOTLY_THEME,
            )
            fig_sil.update_layout(title="Silhouette Score (higher = better)", height=300,
                                   showlegend=False)
            st.plotly_chart(fig_sil, use_container_width=True)

            fig_db = px.bar(
                cdf2.dropna(subset=["Davies-Bouldin"]),
                x="Algorithm", y="Davies-Bouldin",
                color="Algorithm",
                color_discrete_sequence=PALETTE,
                **PLOTLY_THEME,
            )
            fig_db.update_layout(title="Davies-Bouldin Score (lower = better)", height=300,
                                  showlegend=False)
            st.plotly_chart(fig_db, use_container_width=True)

        # Recommendations
        st.markdown("#### 🔍 Clustering Recommendations")
        for _, row_ in cdf2.iterrows():
            algo   = row_["Algorithm"]
            sil    = row_["Silhouette"]
            db     = row_["Davies-Bouldin"]
            ch     = row_["Calinski-Harabasz"]
            n_c    = row_["Clusters"]

            if sil is None:
                verdict = "DBSCAN found noise/one cluster — try adjusting ε or min_samples."
                badge, color = "⚠️", "amber"
            elif sil > 0.5:
                verdict = f"Well-defined clusters detected (Silhouette={sil})."
                badge, color = "✅", "green"
            elif sil > 0.25:
                verdict = f"Moderate cluster structure (Silhouette={sil}). Results may vary."
                badge, color = "🟡", "amber"
            else:
                verdict = f"Weak cluster structure (Silhouette={sil}). Consider different features or k."
                badge, color = "❌", "red"

            st.markdown(f"""
            <div class="ds-card">
              <b>{algo}</b> &nbsp;<span class="ds-badge ds-badge-{color}">{n_c} clusters</span><br>
              {badge} {verdict}<br>
              <span style="color:#64748b;font-size:12px;">
                Silhouette={sil}  |  Davies-Bouldin={db}  |  Calinski-Harabasz={ch}
              </span>
            </div>""", unsafe_allow_html=True)

    # Download comparison report
    st.markdown("---")
    st.markdown("#### ⬇ Download Comparison Report")
    report_lines = ["DatoScope Model Comparison Report\n" + "="*40 + "\n"]
    if reg_res:
        report_lines.append("\nREGRESSION RESULTS\n" + "-"*30)
        for name, r in reg_res.items():
            report_lines.append(f"\n{name}\n  R²={r['R²']}  CV R²={r['CV R²']}  RMSE={r['RMSE']}  MAE={r['MAE']}")
    if clust_res:
        report_lines.append("\n\nCLUSTERING RESULTS\n" + "-"*30)
        for name, r in clust_res.items():
            report_lines.append(
                f"\n{name}\n  Clusters={r['n_clusters']}  Silhouette={r.get('Silhouette')}  "
                f"Davies-Bouldin={r.get('Davies-Bouldin')}  Calinski-Harabasz={r.get('Calinski-Harabasz')}"
            )
    st.download_button(
        "⬇ Download .txt report",
        "\n".join(report_lines).encode(),
        file_name="datascope_comparison_report.txt",
        mime="text/plain",
    )