from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from utils.app_state import active_test_df, active_train_df, init_state
from utils.data_input import render_data_sidebar
from utils.ui import PLOTLY_THEME, setup_page


setup_page("DatoScope · EDA")
init_state()
render_data_sidebar()

st.title("Exploratory Data Analysis")

if st.session_state.train_df is None:
    st.info("Load or generate a dataset first.")
    st.stop()

dataset_choice = st.selectbox("Analyze dataset", ["Train"] + (["Test"] if active_test_df() is not None else []))
df_eda = active_train_df() if dataset_choice == "Train" else active_test_df()
if df_eda is None:
    st.warning("Selected dataset is unavailable.")
    st.stop()

num_cols = df_eda.select_dtypes(include="number").columns.tolist()
if not num_cols:
    st.warning("No numeric columns found.")
    st.stop()

st.markdown("#### Summary Statistics")
stats_df = df_eda[num_cols].describe().T
stats_df["skewness"] = df_eda[num_cols].skew().round(3)
stats_df["kurtosis"] = df_eda[num_cols].kurtosis().round(3)
st.dataframe(stats_df.style.background_gradient(cmap="Blues", subset=["mean", "std"]), use_container_width=True)

st.markdown("---")
st.markdown("#### Feature Distributions")
sel_cols = st.multiselect("Choose features", num_cols, default=num_cols[: min(4, len(num_cols))])
if sel_cols:
    n = len(sel_cols)
    nc = min(3, n)
    nr = (n + nc - 1) // nc
    fig = make_subplots(rows=nr, cols=nc, subplot_titles=sel_cols, vertical_spacing=0.12)
    for idx, col in enumerate(sel_cols):
        row, col_idx = divmod(idx, nc)
        fig.add_trace(
            go.Histogram(x=df_eda[col].dropna(), nbinsx=40, name=col, marker_color="#00d4ff", opacity=0.75),
            row + 1,
            col_idx + 1,
        )
    fig.update_layout(showlegend=False, height=300 * nr, **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("#### Q-Q Plots")
st.caption("Q-Q plots compare each feature against a normal distribution. Strong bends away from the diagonal suggest skew, heavy tails, or non-normal behavior.")
qq_cols = st.multiselect("Choose features for Q-Q plots", num_cols, default=num_cols[: min(2, len(num_cols))], key="qq_cols")
if qq_cols:
    qn = len(qq_cols)
    qcols = min(2, qn)
    qrows = (qn + qcols - 1) // qcols
    qq_fig = make_subplots(rows=qrows, cols=qcols, subplot_titles=qq_cols)
    for idx, col in enumerate(qq_cols):
        row, col_idx = divmod(idx, qcols)
        series = df_eda[col].dropna()
        osm, osr = stats.probplot(series, dist="norm", fit=False)
        slope, intercept, _ = stats.probplot(series, dist="norm", fit=True)[1]
        qq_fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color="#00d4ff", size=5), showlegend=False), row + 1, col_idx + 1)
        x_line = np.array([min(osm), max(osm)])
        y_line = slope * x_line + intercept
        qq_fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", line=dict(color="#f59e0b", dash="dash"), showlegend=False), row + 1, col_idx + 1)
    qq_fig.update_layout(height=320 * qrows, **PLOTLY_THEME)
    st.plotly_chart(qq_fig, use_container_width=True)

st.markdown("---")
st.markdown("#### Correlation Matrix")
if len(num_cols) >= 2:
    corr = df_eda[num_cols].corr()
    fig_c = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig_c.update_layout(height=500, **PLOTLY_THEME)
    st.plotly_chart(fig_c, use_container_width=True)

    corr_flat = corr.mask(np.triu(np.ones(corr.shape, dtype=bool))).stack().reset_index()
    corr_flat.columns = ["Feature A", "Feature B", "Correlation"]
    corr_flat["abs"] = corr_flat["Correlation"].abs()
    top = corr_flat.nlargest(10, "abs").drop(columns="abs")
    with st.expander("Top 10 correlated pairs"):
        st.dataframe(top.reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.markdown("#### Scatter Plot")
if len(num_cols) >= 2:
    col1, col2 = st.columns(2)
    x_col = col1.selectbox("X axis", num_cols, index=0, key=f"eda_x_{dataset_choice}")
    y_col = col2.selectbox("Y axis", num_cols, index=min(1, len(num_cols) - 1), key=f"eda_y_{dataset_choice}")
    color_opts = ["— none —"] + df_eda.columns.tolist()
    color_col = st.selectbox("Color by", color_opts, key=f"eda_color_{dataset_choice}")
    fig_s = px.scatter(df_eda, x=x_col, y=y_col, color=None if color_col == "— none —" else color_col, opacity=0.65, trendline="ols")
    fig_s.update_layout(height=420, **PLOTLY_THEME)
    st.plotly_chart(fig_s, use_container_width=True)

st.markdown("---")
st.markdown("#### Feature Variance Ranking")
st.caption("Variance shows how much a feature spreads out. Higher variance means values are more spread and may carry more separating power, while very low variance often means the feature changes very little and may add limited signal.")
var = df_eda[num_cols].var().sort_values(ascending=False)
var_df = var.reset_index()
var_df.columns = ["Feature", "Variance"]
var_df["Interpretation"] = np.where(
    var_df["Variance"] < max(var_df["Variance"].max() * 0.05, 0.01),
    "Low spread",
    np.where(var_df["Variance"] > var_df["Variance"].median(), "High spread", "Moderate spread"),
)
fig_v = px.bar(
    var_df,
    x="Feature",
    y="Variance",
    color="Interpretation",
    text=var_df["Variance"].round(3),
    color_discrete_map={"High spread": "#00d4ff", "Moderate spread": "#7c3aed", "Low spread": "#f59e0b"},
)
fig_v.update_traces(textposition="outside")
fig_v.update_layout(height=360, yaxis_title="Variance (spread of values)", **PLOTLY_THEME)
st.plotly_chart(fig_v, use_container_width=True)
