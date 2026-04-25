from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.app_state import active_test_df, active_train_df, init_state
from utils.data_input import render_data_sidebar
from utils.modeling import export_model_bytes, infer_supervised_task, run_classification_models, run_regression_models
from utils.ui import PLOTLY_THEME, setup_page


setup_page("DatoScope · Supervised")
init_state()
render_data_sidebar()

st.title("Supervised Modeling")

df_train = active_train_df()
df_test = active_test_df()
if df_train is None:
    st.info("Load or generate a dataset first.")
    st.stop()

num_cols = df_train.select_dtypes(include="number").columns.tolist()
if len(df_train.columns) < 2:
    st.warning("Need at least two columns for supervised modeling.")
    st.stop()

target_default = st.session_state.data_meta.get("target_column")
target_options = df_train.columns.tolist()
default_index = target_options.index(target_default) if target_default in target_options else len(target_options) - 1
target_col = st.selectbox("🎯 Target column", target_options, index=max(0, default_index))
st.session_state.data_meta["target_column"] = target_col

suggested_task = infer_supervised_task(df_train, target_col)
task_type = st.radio("Task type", ["Regression", "Classification"], index=0 if suggested_task == "Regression" else 1)
st.caption(f"Suggested task: {suggested_task}")

feature_candidates = [c for c in df_train.columns if c != target_col]
default_features = [c for c in feature_candidates if pd.api.types.is_numeric_dtype(df_train[c])]
features = st.multiselect("Feature columns (X)", feature_candidates, default=default_features[: min(len(default_features), 10)])
test_size = st.slider("Test split %", 10, 40, 20) / 100
cv_folds = st.slider("CV folds", 3, 10, 5)
random_seed = st.number_input("Split seed", min_value=0, value=42, step=1)

if not features:
    st.warning("Select at least one feature column.")
    st.stop()

missing_in_test = [c for c in features if df_test is not None and c not in df_test.columns]
if missing_in_test:
    st.error(f"Uploaded test data is missing features: {', '.join(missing_in_test)}")
    st.stop()
if df_test is not None and target_col not in df_test.columns:
    st.error("Target column is missing from the uploaded test dataset.")
    st.stop()

numeric_features = [c for c in features if pd.api.types.is_numeric_dtype(df_train[c])]
if len(numeric_features) != len(features):
    st.warning("Non-numeric features selected. Current supervised models use numeric features only.")
    features = numeric_features
    if not features:
        st.stop()

if task_type == "Regression":
    col_a, col_b = st.columns(2)
    with col_a:
        run_lr = st.checkbox("Linear Regression", value=True)
        run_ridge = st.checkbox("Ridge", value=True)
        ridge_alpha = st.select_slider("Ridge α", [0.001, 0.01, 0.1, 1, 10, 100], value=1.0)
    with col_b:
        run_lasso = st.checkbox("Lasso", value=True)
        lasso_alpha = st.select_slider("Lasso α", [0.001, 0.01, 0.1, 1, 10, 100], value=0.1)

    if st.button("🚀 Train Regression Models", use_container_width=True):
        st.session_state.reg_results = run_regression_models(
            df_train,
            df_test,
            features=features,
            target_col=target_col,
            test_size=test_size,
            cv_folds=cv_folds,
            random_seed=random_seed,
            run_lr=run_lr,
            run_ridge=run_ridge,
            ridge_alpha=ridge_alpha,
            run_lasso=run_lasso,
            lasso_alpha=lasso_alpha,
        )
        if st.session_state.reg_results:
            st.session_state.data_meta["split_method"] = next(iter(st.session_state.reg_results.values()))["split_method"]

    res = st.session_state.reg_results
    if not res:
        st.info("Configure and run regression models.")
        st.stop()

    st.caption(f"Evaluation split: {next(iter(res.values()))['split_method']}")
    metric_rows = [{"Model": name, "R²": r["R²"], "CV R²": r["CV R²"], "RMSE": r["RMSE"], "MAE": r["MAE"], "MSE": r["MSE"]} for name, r in res.items()]
    mdf = pd.DataFrame(metric_rows).set_index("Model")
    st.dataframe(
        mdf.style.highlight_max(subset=["R²", "CV R²"], color="#10b98133")
        .highlight_min(subset=["RMSE", "MAE", "MSE"], color="#10b98133")
        .format(precision=4),
        use_container_width=True,
    )

    for name, r in res.items():
        st.markdown(f"---\n#### {name}")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²", r["R²"])
        m2.metric("RMSE", r["RMSE"])
        m3.metric("CV R²", r["CV R²"])
        ytest_arr = np.array(r["y_test"])
        ypred_arr = np.array(r["y_pred"])
        residuals = ytest_arr - ypred_arr
        min_v, max_v = float(ytest_arr.min()), float(ytest_arr.max())
        fa, fb, fc = st.columns(3)

        with fa:
            fig_ap = go.Figure()
            fig_ap.add_trace(go.Scatter(x=ytest_arr, y=ypred_arr, mode="markers", marker=dict(color="#00d4ff", opacity=0.6, size=5)))
            fig_ap.add_trace(go.Scatter(x=[min_v, max_v], y=[min_v, max_v], mode="lines", line=dict(color="#7c3aed", dash="dash")))
            fig_ap.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted", height=300, **PLOTLY_THEME)
            st.plotly_chart(fig_ap, use_container_width=True)

        with fb:
            fig_res = go.Figure()
            fig_res.add_trace(go.Histogram(x=residuals, nbinsx=30, marker_color="#7c3aed", opacity=0.8))
            fig_res.update_layout(title="Residuals", xaxis_title="Residual", yaxis_title="Count", height=300, **PLOTLY_THEME)
            st.plotly_chart(fig_res, use_container_width=True)

        with fc:
            if "coef" in r:
                coef_df = pd.DataFrame({"Feature": list(r["coef"].keys()), "Coefficient": list(r["coef"].values())})
                coef_df = coef_df.reindex(coef_df["Coefficient"].abs().sort_values(ascending=True).index)
                fig_c2 = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", color="Coefficient", color_continuous_scale="RdBu", color_continuous_midpoint=0)
                fig_c2.update_layout(title="Coefficients", height=300, coloraxis_showscale=False, **PLOTLY_THEME)
                st.plotly_chart(fig_c2, use_container_width=True)

        st.download_button(
            f"⬇ Download {name} model",
            export_model_bytes(r["model"], features=r["features"], target=r["target"], task_type="regression"),
            file_name=f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('α', 'alpha')}.pkl",
            mime="application/octet-stream",
            key=f"download_reg_{name}",
        )

else:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        run_logreg = st.checkbox("Logistic Regression", value=True)
    with col_b:
        run_rf = st.checkbox("Random Forest", value=True)
        rf_estimators = st.slider("RF estimators", 50, 500, 200, step=50)
    with col_c:
        run_knn = st.checkbox("KNN", value=False)
        knn_neighbors = st.slider("KNN neighbors", 1, 25, 5)

    if st.button("🚀 Train Classification Models", use_container_width=True):
        st.session_state.cls_results = run_classification_models(
            df_train,
            df_test,
            features=features,
            target_col=target_col,
            test_size=test_size,
            cv_folds=cv_folds,
            random_seed=random_seed,
            run_logreg=run_logreg,
            run_rf=run_rf,
            rf_estimators=rf_estimators,
            run_knn=run_knn,
            knn_neighbors=knn_neighbors,
        )
        if st.session_state.cls_results:
            st.session_state.data_meta["split_method"] = next(iter(st.session_state.cls_results.values()))["split_method"]

    res = st.session_state.cls_results
    if not res:
        st.info("Configure and run classification models.")
        st.stop()

    st.caption(f"Evaluation split: {next(iter(res.values()))['split_method']}")
    metric_rows = [{"Model": name, "Accuracy": r["Accuracy"], "Precision": r["Precision"], "Recall": r["Recall"], "F1": r["F1"], "CV Accuracy": r["CV Accuracy"]} for name, r in res.items()]
    cdf = pd.DataFrame(metric_rows).set_index("Model")
    st.dataframe(
        cdf.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1", "CV Accuracy"], color="#10b98133").format(precision=4),
        use_container_width=True,
    )

    for name, r in res.items():
        st.markdown(f"---\n#### {name}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", r["Accuracy"])
        m2.metric("Precision", r["Precision"])
        m3.metric("Recall", r["Recall"])
        m4.metric("F1", r["F1"])

        cm = r["Confusion Matrix"]
        labels = sorted(pd.Series(r["y_test"]).astype(str).unique())
        fig_cm = px.imshow(cm, text_auto=True, x=labels, y=labels, color_continuous_scale="Blues")
        fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", height=380, **PLOTLY_THEME)
        st.plotly_chart(fig_cm, use_container_width=True)

        report_df = pd.DataFrame(r["Report"]).T
        st.dataframe(report_df, use_container_width=True)

        st.download_button(
            f"⬇ Download {name} model",
            export_model_bytes(r["model"], features=r["features"], target=r["target"], task_type="classification"),
            file_name=f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl",
            mime="application/octet-stream",
            key=f"download_cls_{name}",
        )
