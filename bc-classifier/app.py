import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report, roc_curve)

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}
TARGET = "diagnosis"
LABELS = ["Benign (0)", "Malignant (1)"]


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_data
def load_features():
    return json.load(open("model/feature_names.json"))


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def normalise_target(s):
    """Accept M/B letters or 1/0 numbers."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    return s.astype(str).str.strip().str.upper().map({"M": 1, "B": 0})


# ---------------- Sidebar ----------------
st.sidebar.title("Controls")
uploaded = st.sidebar.file_uploader("Upload TEST data (CSV)", type="csv")
model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))
show_all = st.sidebar.checkbox("Compare all 5 models", value=True)
st.sidebar.caption("The CSV must contain all 30 feature columns and the `diagnosis` column (M/B or 1/0).")

# ---------------- Main ----------------
st.title("Breast Cancer Classification (WDBC)")
st.write("Predict whether a tumour is **benign** or **malignant** from 30 cell-nucleus "
         "measurements using five pre-trained ML models. Models are loaded from disk, "
         "no retraining happens on the uploaded data.")

if uploaded is None:
    st.info(" Upload a test CSV from the sidebar to begin. You can use `test_data.csv` from the GitHub repo.")
    st.stop()

data = pd.read_csv(uploaded)
features = load_features()
missing = [c for c in features + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Missing columns in uploaded file: {missing}")
    st.stop()

X = data[features]
y = normalise_target(data[TARGET])
if y.isna().any():
    st.error("Target column has values other than M/B or 1/0.")
    st.stop()
if y.nunique() < 2:
    st.error("Target column needs both classes to compute AUC/MCC.")
    st.stop()

st.subheader("Uploaded data preview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", len(data))
c2.metric("Benign", int((y == 0).sum()))
c3.metric("Malignant", int((y == 1).sum()))
st.dataframe(data.head(), use_container_width=True)

# ---- Selected model ----
model = load_model(MODEL_FILES[model_name])
pred = model.predict(X)
prob = model.predict_proba(X)[:, 1]
m = compute_metrics(y, pred, prob)

st.subheader(f" Evaluation metrics: {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Purples",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**ROC curve**")
    fpr, tpr, _ = roc_curve(y, prob)
    fig2, ax2 = plt.subplots(figsize=(4.5, 3.6))
    ax2.plot(fpr, tpr, label=f"AUC = {m['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], "--", color="grey")
    ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
    ax2.legend()
    st.pyplot(fig2)

st.markdown("**Classification report**")
report = classification_report(y, pred, target_names=LABELS, output_dict=True)
st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

# ---- Comparison of all models ----
if show_all:
    st.subheader("Comparison of all models on the uploaded test data")
    rows = []
    for name, path in MODEL_FILES.items():
        mdl = load_model(path)
        rows.append({"Model": name, **compute_metrics(y, mdl.predict(X), mdl.predict_proba(X)[:, 1])})
    comp = pd.DataFrame(rows).set_index("Model").round(4)
    st.dataframe(comp.style.highlight_max(axis=0, color="#c7f0d8"), use_container_width=True)
    st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
    st.success(f"Best model by MCC: **{comp['MCC'].idxmax()}**")

# ---- Predictions download ----
out = data.copy()
out["predicted"] = np.where(pred == 1, "M", "B")
out["malignant_probability"] = prob.round(4)
st.download_button("Download predictions CSV", out.to_csv(index=False), "predictions.csv", "text/csv")
