from __future__ import annotations

import matplotlib
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans

from utils.app_state import active_test_df, active_train_df, init_state
from utils.data_input import render_data_sidebar
from utils.modeling import run_clustering_models
from utils.ui import PLOTLY_THEME, PALETTE, setup_page

matplotlib.use("Agg")
import matplotlib.pyplot as plt


setup_page("DatoScope · Clustering")
init_state()
render_data_sidebar()

st.title("Clustering")
df_c = active_train_df()
if df_c is None:
    st.info("Load or generate a dataset first.")
    st.stop()
if active_test_df() is not None:
    st.caption("Clustering runs on the train dataset only. Uploaded test data is kept for supervised workflows.")

num_cols = df_c.select_dtypes(include="number").columns.tolist()
if len(num_cols) < 2:
    st.warning("Need at least 2 numeric columns for clustering.")
    st.stop()

ca2, cb2 = st.columns([1, 2])
with ca2:
    clust_feats = st.multiselect("Feature columns", num_cols, default=num_cols[: min(5, len(num_cols))], key="clust_feats")
    run_km = st.checkbox("K-Means", value=True)
    k_val = st.slider("K (clusters)", 2, 10, 3) if run_km else 3
    run_db = st.checkbox("DBSCAN", value=True)
    db_eps = st.number_input("ε (eps)", min_value=0.01, max_value=10.0, value=0.5, step=0.05) if run_db else 0.5
    db_min = st.slider("min_samples", 2, 20, 5) if run_db else 5
    run_hc = st.checkbox("Hierarchical", value=True)
    hc_k = st.slider("n_clusters", 2, 10, 3, key="hc_k") if run_hc else 3
    hc_link = st.selectbox("Linkage", ["ward", "complete", "average", "single"]) if run_hc else "ward"
    clust_btn = st.button("🚀 Run Clustering", use_container_width=True)

with cb2:
    if not clust_feats:
        st.warning("Select at least 2 feature columns.")
        st.stop()
    X_c = df_c[clust_feats].dropna()
    if len(X_c) < 3:
        st.warning("Not enough valid rows for clustering after removing missing values.")
        st.stop()

    if clust_btn:
        st.session_state.cluster_results = run_clustering_models(
            df_c,
            features=clust_feats,
            run_km=run_km,
            k_val=k_val,
            run_db=run_db,
            db_eps=db_eps,
            db_min=db_min,
            run_hc=run_hc,
            hc_k=hc_k,
            hc_link=hc_link,
        )

    cres = st.session_state.cluster_results
    if not cres:
        st.info("Configure and click Run Clustering to start.")
        st.stop()

    results_c = cres["results"]
    X_plot = cres["X"]
    feats_c = cres["features"]

    metric_rows_c = [
        {
            "Algorithm": name,
            "Clusters found": r["n_clusters"],
            "Silhouette ↑": r.get("Silhouette"),
            "Davies-Bouldin ↓": r.get("Davies-Bouldin"),
            "Calinski-Harabasz ↑": r.get("Calinski-Harabasz"),
        }
        for name, r in results_c.items()
    ]
    cmdf = pd.DataFrame(metric_rows_c).set_index("Algorithm")
    st.dataframe(
        cmdf.style.highlight_max(subset=["Silhouette ↑", "Calinski-Harabasz ↑"], color="#10b98133")
        .highlight_min(subset=["Davies-Bouldin ↓"], color="#10b98133")
        .format(na_rep="N/A", precision=4),
        use_container_width=True,
    )

    if X_plot.shape[1] > 2:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_plot)
        xlab = f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
        ylab = f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
    else:
        coords = X_plot.values
        xlab, ylab = feats_c[0], feats_c[1]

    for name, r in results_c.items():
        st.markdown(f"---\n#### {name} — {r['n_clusters']} cluster(s)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clusters", r["n_clusters"])
        m2.metric("Silhouette", r.get("Silhouette") or "N/A")
        m3.metric("Davies-Bouldin", r.get("Davies-Bouldin") or "N/A")
        m4.metric("Calinski-Harabasz", r.get("Calinski-Harabasz") or "N/A")

        labels_arr = r["labels"]
        plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "Cluster": labels_arr.astype(str)})
        fig_cl = px.scatter(plot_df, x="x", y="y", color="Cluster", color_discrete_sequence=PALETTE, labels={"x": xlab, "y": ylab}, opacity=0.75)
        fig_cl.update_layout(height=380, legend_title_text="Cluster", title=f"{name} — Cluster Assignments", **PLOTLY_THEME)
        fig_cl.update_traces(marker_size=6)
        st.plotly_chart(fig_cl, use_container_width=True)

        sizes = pd.Series(labels_arr).value_counts().sort_index()
        fig_sz = px.bar(x=sizes.index.astype(str), y=sizes.values, labels={"x": "Cluster", "y": "Count"}, color=sizes.values, color_continuous_scale="plasma")
        fig_sz.update_layout(title="Cluster sizes", coloraxis_showscale=False, height=260, **PLOTLY_THEME)
        st.plotly_chart(fig_sz, use_container_width=True)

        if name == "Hierarchical":
            st.markdown("##### Dendrogram (top 30 leaves)")
            fig_dend, ax = plt.subplots(figsize=(14, 4))
            fig_dend.patch.set_facecolor("#111827")
            ax.set_facecolor("#1a2235")
            Z = linkage(X_plot, method=hc_link)
            dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, color_threshold=0.6 * max(Z[:, 2]), above_threshold_color="#64748b")
            ax.tick_params(colors="#e2e8f0")
            for spine in ax.spines.values():
                spine.set_color("#1e3a5f")
            plt.tight_layout()
            st.pyplot(fig_dend, use_container_width=True)
            plt.close()

        if name == "K-Means":
            st.markdown("##### Elbow Curve (inertia)")
            inertias = []
            k_range = range(2, min(11, len(X_plot)))
            for ki in k_range:
                km = KMeans(n_clusters=ki, random_state=42, n_init=10)
                km.fit(X_plot)
                inertias.append(km.inertia_)
            fig_elbow = px.line(x=list(k_range), y=inertias, labels={"x": "k", "y": "Inertia"}, markers=True)
            fig_elbow.update_traces(line_color="#00d4ff", marker_color="#7c3aed", marker_size=8)
            fig_elbow.update_layout(title="Elbow Method", height=280, **PLOTLY_THEME)
            st.plotly_chart(fig_elbow, use_container_width=True)
