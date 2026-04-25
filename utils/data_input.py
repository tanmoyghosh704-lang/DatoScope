"""
Shared sidebar data-input controls for the DatoScope app.
"""

from __future__ import annotations

import streamlit as st

from utils.app_state import reset_model_results, set_data
from utils.generators import CLUSTERING_DATASETS, REGRESSION_DATASETS, generate_dataset
from utils.preprocessing import clean_datasets, load_uploaded_file
from utils.ui import render_sidebar_brand


def render_data_sidebar() -> None:
    with st.sidebar:
        render_sidebar_brand()

        data_mode = st.radio(
            "Data source",
            ["Generate Dataset", "Upload Single File", "Upload Train/Test"],
            index=["Generate Dataset", "Upload Single File", "Upload Train/Test"].index(st.session_state.data_source_mode)
            if st.session_state.data_source_mode in ["Generate Dataset", "Upload Single File", "Upload Train/Test"]
            else 1,
        )

        if data_mode == "Generate Dataset":
            task_type = st.selectbox("Task type", ["Regression", "Clustering"])
            dataset_options = REGRESSION_DATASETS if task_type == "Regression" else CLUSTERING_DATASETS
            dataset_type = st.selectbox("Dataset type", dataset_options)
            n_samples = st.slider("Samples", 100, 5000, 500, step=50)
            noise = st.slider("Noise", 0.01, 2.0, 0.15, step=0.01)
            n_clusters = st.slider("Clusters / arms", 2, 10, 3) if task_type == "Clustering" else 3
            min_features = 2 if task_type == "Clustering" else 1
            default_features = 20 if dataset_type == "High-Dimensional" else max(min_features, 6)
            n_features = st.slider("Number of features", min_features, 50, default_features)
            n_informative = st.slider("Informative features", 2, n_features, min(5, n_features)) if dataset_type == "High-Dimensional" else 5
            random_seed = st.number_input("Random seed", min_value=0, value=42, step=1)

            if st.button("🧪 Generate Dataset", use_container_width=True):
                df_gen, meta = generate_dataset(
                    task_type,
                    dataset_type,
                    n_samples=n_samples,
                    noise=noise,
                    n_clusters=n_clusters,
                    random_seed=random_seed,
                    n_features=n_features,
                    n_informative=n_informative,
                )
                meta["split_method"] = "Auto split from generated train data"
                set_data(
                    train_df=df_gen,
                    test_df=None,
                    raw_df=df_gen,
                    train_filename=f"{task_type.lower()}_{dataset_type.lower().replace(' ', '_')}.csv",
                    source_mode="Generate Dataset",
                    metadata=meta,
                )
                st.success(f"Generated {dataset_type} dataset")

        elif data_mode == "Upload Single File":
            uploaded = st.file_uploader("Upload dataset", type=["csv", "xls", "xlsx", "zip", "data"], key="single_upload")
            if uploaded:
                df_raw = load_uploaded_file(uploaded)
                set_data(
                    train_df=df_raw,
                    test_df=None,
                    raw_df=df_raw,
                    train_filename=uploaded.name,
                    source_mode="Upload Single File",
                    metadata={"source": "uploaded", "split_method": "Auto split from uploaded train file", "dataset_type": "User upload"},
                )
                st.success(f"Loaded {uploaded.name}")
                st.caption(f"{df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

        else:
            train_upload = st.file_uploader("Train file", type=["csv", "xls", "xlsx", "zip", "data"], key="train_upload")
            test_upload = st.file_uploader("Test file (optional)", type=["csv", "xls", "xlsx", "zip", "data"], key="test_upload")
            if train_upload:
                train_df = load_uploaded_file(train_upload)
                test_df = load_uploaded_file(test_upload) if test_upload else None
                if test_df is not None and list(train_df.columns) != list(test_df.columns):
                    st.error("Train/test column mismatch. Upload files with the same schema.")
                else:
                    set_data(
                        train_df=train_df,
                        test_df=test_df,
                        raw_df=train_df,
                        train_filename=train_upload.name,
                        test_filename=test_upload.name if test_upload else "",
                        source_mode="Upload Train/Test",
                        metadata={
                            "source": "uploaded",
                            "split_method": "Uploaded test file" if test_df is not None else "Auto split from uploaded train file",
                            "dataset_type": "User upload",
                        },
                    )
                    st.success("Train/test files loaded" if test_df is not None else "Train file loaded")

        st.divider()

        if st.session_state.train_df is not None:
            st.markdown("#### Preprocessing")
            missing_strat = st.selectbox("Missing values", ["mean", "median", "mode", "drop"])
            outlier_meth = st.selectbox("Outlier method", ["IQR", "Z-Score", "None"])
            scale_meth = st.selectbox("Scaler", ["Standard", "MinMax", "Robust"])
            remove_dupes = st.checkbox("Remove duplicates", value=True)
            train_cols = st.session_state.train_df.columns.tolist()
            label_col = st.selectbox("Label / target column (optional — keeps it unscaled)", ["— none —"] + train_cols)
            label_col = None if label_col == "— none —" else label_col
            st.session_state.data_meta["target_column"] = label_col or "Not selected yet"

            if st.button("⚙️ Clean & Preprocess", use_container_width=True):
                with st.spinner("Cleaning datasets…"):
                    train_clean, test_clean, train_report, test_report = clean_datasets(
                        st.session_state.train_df,
                        st.session_state.test_df,
                        missing_strategy=missing_strat,
                        outlier_method=outlier_meth if outlier_meth != "None" else "none",
                        scale_method=scale_meth,
                        remove_dupes=remove_dupes,
                        label_col=label_col,
                    )
                    st.session_state.clean_df = train_clean
                    st.session_state.clean_train_df = train_clean
                    st.session_state.clean_test_df = test_clean
                    st.session_state.clean_report = train_report
                    st.session_state.clean_report_train = train_report
                    st.session_state.clean_report_test = test_report
                    reset_model_results()
                st.success("Dataset preprocessing complete")

        st.divider()
        st.caption("DatoScope · MA25 Project")
