"""
DatoScope official Streamlit entrypoint.

This page focuses on data input and dataset overview. Other workflows live in
the Streamlit pages/ directory.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from utils.app_state import init_state
from utils.data_input import render_data_sidebar
from utils.ui import PLOTLY_THEME, dataset_preview_metrics, render_metadata_panel, setup_page, show_dataset_block


setup_page("DatoScope")
init_state()
render_data_sidebar()

st.title("DatoScope")
st.caption("Generate or upload data, clean it, then move through the pages in the sidebar for EDA and modeling.")

if st.session_state.train_df is None:
    st.info("Use the sidebar to generate a dataset or upload train/test files.")
    st.stop()

render_metadata_panel(st.session_state.data_meta, st.session_state.train_df, st.session_state.test_df)
dataset_preview_metrics(st.session_state.train_df, "Train")
show_dataset_block("Train Dataset", st.session_state.train_df)

if st.session_state.data_meta.get("source") == "generated":
    st.markdown("---")
    st.markdown("#### Generated Dataset Plot")
    generated_df = st.session_state.train_df.copy()
    numeric_cols = generated_df.select_dtypes(include="number").columns.tolist()
    target_col = st.session_state.data_meta.get("target_column")
    color_col = None
    for candidate in [target_col, "label", "target"]:
        if candidate and candidate in generated_df.columns:
            color_col = candidate
            break

    if len(numeric_cols) >= 2:
        plot_features = [col for col in numeric_cols if col != color_col]
        if len(plot_features) >= 2:
            x_col, y_col = plot_features[:2]
        else:
            x_col, y_col = numeric_cols[:2]
        fig_generated = px.scatter(
            generated_df,
            x=x_col,
            y=y_col,
            color=color_col,
            opacity=0.8,
            title=f"Generated data preview: {x_col} vs {y_col}",
        )
        fig_generated.update_traces(marker=dict(size=9, line=dict(width=0.5, color="#111827")))
        fig_generated.update_layout(height=430, **PLOTLY_THEME)
        st.plotly_chart(fig_generated, use_container_width=True)
        if color_col:
            st.caption(
                f"This preview uses the first two numeric features and colors points by `{color_col}` so you can quickly inspect the generated structure."
            )
        else:
            st.caption("This preview uses the first two numeric features so you can quickly inspect the generated structure.")
    elif len(numeric_cols) == 1:
        fig_generated = px.histogram(
            generated_df,
            x=numeric_cols[0],
            color=color_col,
            nbins=40,
            title=f"Generated data preview: distribution of {numeric_cols[0]}",
        )
        fig_generated.update_layout(height=380, **PLOTLY_THEME)
        st.plotly_chart(fig_generated, use_container_width=True)
        st.caption("Only one numeric feature is available, so the preview is shown as a distribution plot.")
    else:
        st.info("A generated-data preview plot is unavailable because no numeric feature columns were found.")

if st.session_state.test_df is not None:
    st.markdown("---")
    dataset_preview_metrics(st.session_state.test_df, "Test")
    show_dataset_block("Test Dataset", st.session_state.test_df)

if st.session_state.clean_train_df is not None:
    st.markdown("---")
    st.markdown(
        """
        <div class="cleaned-preview-banner">
          <b>Cleaned Dataset Preview</b><br>
          Your processed train dataset is ready for EDA, supervised modeling, and clustering.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("#### Cleaned Train Preview")
    report = st.session_state.clean_report_train or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows after cleaning", f"{report.get('rows_out', len(st.session_state.clean_train_df)):,}")
    c2.metric("Columns after cleaning", f"{report.get('cols_out', st.session_state.clean_train_df.shape[1])}")
    c3.metric("Actions taken", f"{len(report.get('actions', []))}")
    if report.get("actions"):
        with st.expander("Train cleaning log"):
            for action in report["actions"]:
                st.write(f"• {action}")
    st.dataframe(st.session_state.clean_train_df.head(200), use_container_width=True, height=260)
    st.download_button(
        "⬇ Download cleaned train CSV",
        st.session_state.clean_train_df.to_csv(index=False).encode(),
        file_name="datascope_clean_train.csv",
        mime="text/csv",
    )

    if st.session_state.clean_test_df is not None:
        st.markdown("#### Cleaned Test Preview")
        test_report = st.session_state.clean_report_test or {}
        if test_report.get("actions"):
            with st.expander("Test cleaning log"):
                for action in test_report["actions"]:
                    st.write(f"• {action}")
        st.dataframe(st.session_state.clean_test_df.head(200), use_container_width=True, height=240)
        st.download_button(
            "⬇ Download cleaned test CSV",
            st.session_state.clean_test_df.to_csv(index=False).encode(),
            file_name="datascope_clean_test.csv",
            mime="text/csv",
        )

st.markdown("---")
st.markdown("#### Missing Values (Train)")
miss = st.session_state.train_df.isnull().mean() * 100
miss = miss[miss > 0].sort_values(ascending=False)
if miss.empty:
    st.success("No missing values in the train dataset")
else:
    fig = px.bar(
        x=miss.index,
        y=miss.values,
        labels={"x": "Column", "y": "Missing %"},
        color=miss.values,
        color_continuous_scale="plasma",
    )
    fig.update_layout(coloraxis_showscale=False, height=320, **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)
