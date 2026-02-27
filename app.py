import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.impute import SimpleImputer

# Classification models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Trainer", layout="wide")
st.title("ML Trainer - Problem Type → Filtered Models + Targets")

# -----------------------------
# Helpers
# -----------------------------
def is_probably_id_column(s: pd.Series) -> bool:
    s2 = s.dropna()
    if len(s2) == 0:
        return False
    return (s2.nunique() / len(s2)) > 0.98

def get_categorical_columns(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_bool_dtype(s):
            cols.append(c)
        else:
            if pd.api.types.is_numeric_dtype(s) and s.nunique() <= 20:
                cols.append(c)
    return cols

def get_numeric_columns(df: pd.DataFrame) -> list:
    cols = []
    for c in df.columns:
        s = df[c].dropna()
        if len(s) == 0:
            continue
        if pd.api.types.is_numeric_dtype(s):
            cols.append(c)
    return cols

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )

# -----------------------------
# 1) Input Dataset
# -----------------------------
st.subheader("1) Input Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error("Could not read the CSV. Make sure it's a valid CSV file.")
    st.write("Error:", str(e))
    st.stop()

if df.shape[1] < 2:
    st.error("Dataset must have at least 2 columns.")
    st.stop()

st.write("Dataset Preview:")
st.dataframe(df.head(10), use_container_width=True)

# -----------------------------
# 2) Problem Type
# -----------------------------
st.subheader("2) Select Problem Type")
problem_type = st.radio(
    "Choose type:",
    options=["Classification", "Regression (Numerical)"],
    horizontal=True
)
is_classification = (problem_type == "Classification")

# -----------------------------
# 3) Model selection
# -----------------------------
st.subheader("3) Select Model")

classification_models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=200, random_state=42)
}

regression_models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42)
}

if is_classification:
    model_name = st.selectbox("Choose Classification Model", list(classification_models.keys()))
    estimator = classification_models[model_name]
else:
    model_name = st.selectbox("Choose Regression Model", list(regression_models.keys()))
    estimator = regression_models[model_name]

# -----------------------------
# 4) Target variable filtered
# -----------------------------
st.subheader("4) Target Variable")

categorical_targets = get_categorical_columns(df)
numeric_targets = get_numeric_columns(df)

if is_classification:
    target_options = categorical_targets
    if len(target_options) == 0:
        st.error("No categorical target columns found for Classification.")
        st.stop()
else:
    target_options = numeric_targets
    if len(target_options) == 0:
        st.error("No numeric target columns found for Regression.")
        st.stop()

target_col = st.selectbox("Select Target Column", options=target_options)

# -----------------------------
# 5) Features
# -----------------------------
st.subheader("5) Features")

candidate_features = [c for c in df.columns if c != target_col]
default_features = [c for c in candidate_features if not is_probably_id_column(df[c])]
if len(default_features) == 0:
    default_features = candidate_features

feature_cols = st.multiselect(
    "Select Feature Columns",
    options=candidate_features,
    default=default_features
)

if len(feature_cols) == 0:
    st.warning("Select at least 1 feature column.")
    st.stop()

# -----------------------------
# 6) Train-Test Split
# -----------------------------
st.subheader("6) Train-Test Split")
test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5) / 100.0

# -----------------------------
# 7) Evaluate Button
# -----------------------------
st.subheader("7) Evaluate Model")
evaluate = st.button("Evaluate Model")

if evaluate:
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    if y.isna().any():
        st.warning("Target has missing values. Dropping rows with missing target.")
        mask = ~y.isna()
        X = X.loc[mask]
        y = y.loc[mask]

    if len(y) < 10:
        st.error("Not enough data after cleaning.")
        st.stop()

    stratify_arg = y if (is_classification and y.nunique() > 1) else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify_arg
        )
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    preprocessor = build_preprocessor(X)

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", estimator)
    ])

    try:
        model.fit(X_train, y_train)
    except Exception as e:
        st.error("Training failed. Check your data types and missing values.")
        st.write("Error:", str(e))
        st.stop()

    y_pred = model.predict(X_test)

    st.success("Model evaluated successfully!")

    # -----------------------------
    # Outputs
    # -----------------------------
    if is_classification:
        acc = accuracy_score(y_test, y_pred)
        labels_sorted = np.unique(y_test)
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.write(f"**Model:** {model_name}")
            st.write(f"**Accuracy:** {acc:.4f}")

            st.write("**Classification Report:**")
            st.text(classification_report(y_test, y_pred, zero_division=0))

            st.write("### Macro-Average Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Macro Precision", f"{precision_macro:.4f}")
            m2.metric("Macro Recall", f"{recall_macro:.4f}")
            m3.metric("Macro F1-Score", f"{f1_macro:.4f}")

        with col2:
            st.write("**Confusion Matrix:**")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.imshow(cm)

            ax.set_xticks(range(len(labels_sorted)))
            ax.set_yticks(range(len(labels_sorted)))
            ax.set_xticklabels([str(x) for x in labels_sorted], rotation=45, ha="right")
            ax.set_yticklabels([str(x) for x in labels_sorted])

            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)

            st.pyplot(fig, use_container_width=True)

        # =============================
        # SIDE-BY-SIDE MACRO GRAPHS
        # =============================
        st.write("### Macro Graphs")

        g1, g2 = st.columns(2)

        # LEFT: Macro Metrics Bar
        with g1:
            st.write("**Macro Precision / Recall / F1**")
            fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
            metric_names = ["Precision", "Recall", "F1"]
            metric_values = [precision_macro, recall_macro, f1_macro]

            ax_bar.bar(metric_names, metric_values)
            ax_bar.set_ylim(0, 1)
            ax_bar.set_ylabel("Score")
            ax_bar.set_title("Macro Metrics")

            for i, v in enumerate(metric_values):
                ax_bar.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

            st.pyplot(fig_bar, use_container_width=True)

        # RIGHT: Macro ROC Curve
        with g2:
            st.write("**Macro ROC Curve**")

            if hasattr(model.named_steps["model"], "predict_proba"):
                y_score = model.predict_proba(X_test)
                classes = np.unique(y_train)
                n_classes = len(classes)

                fig_roc, ax_roc = plt.subplots(figsize=(5, 3))

                if n_classes == 2:
                    y_test_bin = (y_test == classes[1]).astype(int)
                    fpr, tpr, _ = roc_curve(y_test_bin, y_score[:, 1])
                    roc_auc = auc(fpr, tpr)

                    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                    ax_roc.plot([0, 1], [0, 1], linestyle="--")
                    ax_roc.set_title("ROC (Binary)")
                    ax_roc.set_xlabel("FPR")
                    ax_roc.set_ylabel("TPR")
                    ax_roc.legend(loc="lower right")

                else:
                    y_test_bin = label_binarize(y_test, classes=classes)

                    fpr = {}
                    tpr = {}

                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])

                    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                    mean_tpr = np.zeros_like(all_fpr)

                    for i in range(n_classes):
                        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

                    mean_tpr /= n_classes
                    macro_auc = auc(all_fpr, mean_tpr)

                    ax_roc.plot(all_fpr, mean_tpr, label=f"Macro AUC = {macro_auc:.2f}", linewidth=2)
                    ax_roc.plot([0, 1], [0, 1], linestyle="--")
                    ax_roc.set_title("Macro ROC (Multiclass)")
                    ax_roc.set_xlabel("FPR")
                    ax_roc.set_ylabel("TPR")
                    ax_roc.legend(loc="lower right")

                st.pyplot(fig_roc, use_container_width=True)

            else:
                st.warning("Selected model does not support probability scores for ROC.")

    else:
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.write(f"**Model:** {model_name}")
        st.write("### Regression Metrics")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"{mae:.4f}")
        c2.metric("MSE", f"{mse:.4f}")
        c3.metric("RMSE", f"{rmse:.4f}")
        c4.metric("R²", f"{r2:.4f}")

        st.write("### Actual vs Predicted")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_test, y_pred)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig, use_container_width=True)