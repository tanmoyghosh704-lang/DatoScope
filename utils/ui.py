"""
Shared UI helpers and theme for the DatoScope Streamlit app.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st


PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="DM Sans", color="#e2e8f0"),
    margin=dict(l=10, r=10, t=40, b=10),
)
PALETTE = px.colors.qualitative.Bold


def setup_page(title: str = "DatoScope") -> None:
    st.set_page_config(page_title=title, page_icon="🔬", layout="wide", initial_sidebar_state="expanded")
    st.markdown(
        """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

      :root {
        --bg: #0a0e1a; --surface: #111827; --card: #1a2235; --border: #1e3a5f;
        --accent: #00d4ff; --accent2: #7c3aed; --green: #10b981; --amber: #f59e0b;
        --red: #ef4444; --text: #e2e8f0; --muted: #64748b;
      }
      html, body, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important; color: var(--text); font-family: 'DM Sans', sans-serif;
      }
      [data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
      div[data-testid="metric-container"] {
        background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px;
      }
      div[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 12px; }
      div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--accent) !important; font-family: 'Space Mono', monospace; font-size: 1.4rem;
      }
      .stDataFrame, .stTable { border-radius: 10px; overflow: hidden; }
      .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent2)); color: #fff !important;
        border: none; border-radius: 8px; font-weight: 600; font-family: 'DM Sans', sans-serif; padding: 10px 24px;
      }
      .stSelectbox > div > div, .stMultiSelect > div > div, .stSlider > div, .stNumberInput > div, .stRadio > div {
        background: var(--card) !important; border-color: var(--border) !important; color: var(--text) !important;
      }
      h1,h2,h3 { font-family: 'Space Mono', monospace; }
      .ds-card { background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 20px 24px; margin-bottom: 16px; }
      .ds-badge {
        display:inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; letter-spacing: .5px;
      }
      .ds-badge-cyan { background: #00d4ff22; color: #00d4ff; border: 1px solid #00d4ff44; }
      .ds-badge-green { background: #10b98122; color: #10b981; border: 1px solid #10b98144; }
      .ds-badge-amber { background: #f59e0b22; color: #f59e0b; border: 1px solid #f59e0b44; }
      .ds-badge-red { background: #ef444422; color: #ef4444; border: 1px solid #ef444444; }
      .winner-banner {
        background: linear-gradient(135deg, #10b98133, #00d4ff11);
        border: 1px solid #10b981; border-radius: 12px; padding: 16px 20px; text-align: center;
      }
      hr { border-color: var(--border); }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    st.markdown(
        """
    <div style='padding:8px 0 24px'>
      <div style='font-family:Space Mono,monospace;font-size:1.5rem;color:#00d4ff;font-weight:700;letter-spacing:-1px;'>🔬 DatoScope</div>
      <div style='color:#64748b;font-size:12px;margin-top:4px;'>Interactive ML Platform</div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_metadata_panel(meta: dict, train_df: pd.DataFrame | None, test_df: pd.DataFrame | None) -> None:
    source = meta.get("source", "uploaded")
    split_method = meta.get("split_method", "Auto split from train file")
    target_col = meta.get("target_column", "Not selected yet")
    feature_count = meta.get("n_features", train_df.shape[1] if train_df is not None else 0)
    st.markdown("#### Dataset Metadata")
    st.markdown(
        f"""
        <div class="ds-card">
          <b>Source</b>: <span class="ds-badge ds-badge-cyan">{source.upper()}</span><br>
          <b>Train rows</b>: <code>{len(train_df) if train_df is not None else 0}</code><br>
          <b>Test rows</b>: <code>{len(test_df) if test_df is not None else 0}</code><br>
          <b>Feature count</b>: <code>{feature_count}</code><br>
          <b>Target column</b>: <code>{target_col}</code><br>
          <b>Split method</b>: <code>{split_method}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataset_preview_metrics(df: pd.DataFrame, label: str) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{label} rows", f"{df.shape[0]:,}")
    c2.metric(f"{label} cols", f"{df.shape[1]}")
    c3.metric(f"{label} missing", f"{df.isnull().sum().sum():,}")
    c4.metric(f"{label} duplicates", f"{df.duplicated().sum():,}")


def show_dataset_block(title: str, df: pd.DataFrame) -> None:
    st.markdown(f"#### {title}")
    left, right = st.columns(2)
    with left:
        st.dataframe(df.head(200), use_container_width=True, height=320)
    with right:
        summary = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "non-null": df.notnull().sum(),
                "missing%": (df.isnull().mean() * 100).round(2),
                "unique": df.nunique(),
            }
        )
        st.dataframe(summary, use_container_width=True, height=320)
