from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.app_state import init_state
from utils.data_input import render_data_sidebar
from utils.ui import PLOTLY_THEME, setup_page


setup_page("DatoScope · Comparison")
init_state()
render_data_sidebar()

st.title("Model Comparison")

reg_res = st.session_state.reg_results
cls_res = st.session_state.cls_results
clust_res = st.session_state.cluster_results.get("results", {})

if not reg_res and not cls_res and not clust_res:
    st.info("Train supervised or clustering models to compare them here.")
    st.stop()

if reg_res:
    st.markdown("### Regression Models")
    rdf = pd.DataFrame([{"Model": name, "R²": r["R²"], "CV R²": r["CV R²"], "RMSE": r["RMSE"], "MAE": r["MAE"]} for name, r in reg_res.items()])
    best_r = rdf.loc[rdf["R²"].idxmax()]
    st.markdown(
        f"""
        <div class="winner-banner">
          <div style="font-size:11px;color:#10b981;letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:4px;">Best Regression Model</div>
          <div style="font-size:1.4rem;font-weight:700;font-family:'Space Mono',monospace;color:#fff;">🥇 {best_r['Model']}</div>
          <div style="color:#94a3b8;font-size:13px;margin-top:6px;">R² = {best_r['R²']} · CV R² = {best_r['CV R²']} · RMSE = {best_r['RMSE']}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    fig_r = px.bar(rdf, x="Model", y=["R²", "CV R²"], barmode="group", color_discrete_sequence=["#00d4ff", "#7c3aed"])
    fig_r.update_layout(title="R² Comparison", height=340, legend_title_text="Metric", **PLOTLY_THEME)
    st.plotly_chart(fig_r, use_container_width=True)

if cls_res:
    st.markdown("### Classification Models")
    cdf = pd.DataFrame([{"Model": name, "Accuracy": r["Accuracy"], "Precision": r["Precision"], "Recall": r["Recall"], "F1": r["F1"], "CV Accuracy": r["CV Accuracy"]} for name, r in cls_res.items()])
    best_c = cdf.loc[cdf["Accuracy"].idxmax()]
    st.markdown(
        f"""
        <div class="winner-banner">
          <div style="font-size:11px;color:#10b981;letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:4px;">Best Classification Model</div>
          <div style="font-size:1.4rem;font-weight:700;font-family:'Space Mono',monospace;color:#fff;">🥇 {best_c['Model']}</div>
          <div style="color:#94a3b8;font-size:13px;margin-top:6px;">Accuracy = {best_c['Accuracy']} · F1 = {best_c['F1']} · CV Accuracy = {best_c['CV Accuracy']}</div>
        </div>""",
        unsafe_allow_html=True,
    )
    fig_c = px.bar(cdf, x="Model", y=["Accuracy", "F1", "CV Accuracy"], barmode="group")
    fig_c.update_layout(title="Classification Metrics", height=340, **PLOTLY_THEME)
    st.plotly_chart(fig_c, use_container_width=True)

if clust_res:
    st.markdown("### Clustering Models")
    cldf = pd.DataFrame(
        [
            {
                "Algorithm": name,
                "Clusters": r["n_clusters"],
                "Silhouette": r.get("Silhouette"),
                "Davies-Bouldin": r.get("Davies-Bouldin"),
                "Calinski-Harabasz": r.get("Calinski-Harabasz"),
            }
            for name, r in clust_res.items()
        ]
    )
    valid = cldf.dropna(subset=["Silhouette"])
    if not valid.empty:
        best_cluster = valid.loc[valid["Silhouette"].idxmax()]
        st.markdown(
            f"""
            <div class="winner-banner">
              <div style="font-size:11px;color:#10b981;letter-spacing:1px;text-transform:uppercase;font-family:'Space Mono',monospace;margin-bottom:4px;">Best Clustering Algorithm</div>
              <div style="font-size:1.4rem;font-weight:700;font-family:'Space Mono',monospace;color:#fff;">🥇 {best_cluster['Algorithm']}</div>
              <div style="color:#94a3b8;font-size:13px;margin-top:6px;">Silhouette = {best_cluster['Silhouette']} · Davies-Bouldin = {best_cluster['Davies-Bouldin']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.dataframe(cldf, use_container_width=True)

if reg_res:
    st.markdown("### Regression Radar")
    rdf = pd.DataFrame([{"Model": name, "R²": r["R²"], "CV R²": r["CV R²"], "RMSE": r["RMSE"], "MAE": r["MAE"]} for name, r in reg_res.items()])
    radar_df = rdf.copy()
    for col in ["R²", "CV R²"]:
        max_v = radar_df[col].max()
        radar_df[f"{col}_norm"] = radar_df[col] / max_v if max_v else 0
    for col in ["RMSE", "MAE"]:
        max_v = radar_df[col].max()
        radar_df[f"{col}_norm"] = 1 - (radar_df[col] / max_v if max_v else 0)
    cats = ["R²", "CV R²", "RMSE (inv)", "MAE (inv)"]
    fig_rad = go.Figure()
    colors_ = ["#00d4ff", "#7c3aed", "#10b981"]
    for i, (_, row_) in enumerate(radar_df.iterrows()):
        vals = [row_["R²_norm"], row_["CV R²_norm"], row_["RMSE_norm"], row_["MAE_norm"]]
        vals += [vals[0]]
        fig_rad.add_trace(go.Scatterpolar(r=vals, theta=cats + [cats[0]], fill="toself", name=row_["Model"], line_color=colors_[i % len(colors_)], fillcolor=colors_[i % len(colors_)] + "33"))
    fig_rad.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#64748b"), gridcolor="#1e3a5f"), angularaxis=dict(tickfont=dict(color="#e2e8f0"), gridcolor="#1e3a5f"), bgcolor="#111827"),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="DM Sans"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_rad, use_container_width=True)
