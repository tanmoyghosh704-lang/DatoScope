"""
DatoScope main entrypoint.

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

if st.session_state.test_df is not None:
    st.markdown("---")
    dataset_preview_metrics(st.session_state.test_df, "Test")
    show_dataset_block("Test Dataset", st.session_state.test_df)

if st.session_state.clean_train_df is not None:
    st.markdown("---")
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
