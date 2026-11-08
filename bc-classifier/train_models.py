"""Train 5 classifiers on the WDBC breast cancer dataset and save them."""
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

SEED = 42
RAW = "wdbc_converted.csv"

df = pd.read_csv(RAW)
df = df.drop(columns=["id"])                         # id carries no information
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})  # Malignant = positive class

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

# Test data given to the Streamlit app (features + target column)
test_df = X_test.copy()
test_df["diagnosis"] = y_test
test_df.to_csv("test_data.csv", index=False)

models = {
    "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                     ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=SEED),
    "KNN": Pipeline([("sc", StandardScaler()),
                     ("clf", KNeighborsClassifier(n_neighbors=7))]),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([("sc", StandardScaler()),
                                        ("clf", SVC(kernel="rbf", probability=True, random_state=SEED))]),
}

rows = []
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    fname = "model/" + name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(mdl, fname)
    pred = mdl.predict(X_test)
    proba = mdl.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, proba),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "MCC": matthews_corrcoef(y_test, pred),
    })

res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics_summary.csv", index=False)
json.dump(list(X.columns), open("model/feature_names.json", "w"))
print(res.to_string(index=False))
print("\nTrain size:", len(X_train), "Test size:", len(X_test))
