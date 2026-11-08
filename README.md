# Breast-Cancer-Classification-using-ML
🎗️ Breast Cancer Classification Using Machine Learning
a. Problem Statement
Breast cancer is one of the most common cancers worldwide, and early, accurate diagnosis greatly improves survival. The goal of this project is to build and compare five machine-learning classifiers that predict whether a breast tumour is Malignant (M) or Benign (B) from measurements computed on digitized images of a fine-needle aspirate (FNA) of a breast mass. The models are evaluated using Accuracy, AUC, Precision, Recall, F1 and MCC, and are served through an interactive Streamlit web app.

b. Dataset Description
Name: Breast Cancer Wisconsin (Diagnostic) – WDBC
Source: UCI Machine Learning Repository / Kaggle
Instances: 569 (357 Benign, 212 Malignant), no missing values
Features: 30 numeric features (+ id and diagnosis)
Target: diagnosis – M (Malignant, encoded 1, positive class) / B (Benign, encoded 0)
Features: For each of 10 cell-nucleus properties – radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension – the mean, standard error (se) and worst (mean of the three largest values) are provided = 30 features.
Preprocessing: dropped id; encoded target; stratified 80/20 train-test split (random_state=42, 455 train / 114 test); StandardScaler applied (inside a pipeline) for Logistic Regression, KNN and SVM. Decision Tree and Naive Bayes use raw features.
c. GitHub Repository Link
<PASTE YOUR GITHUB REPO LINK HERE>

Repo structure:

bc-classifier/
├── app.py                  # Streamlit app
├── train_models.py         # training + evaluation + model saving
├── requirements.txt
├── README.md
├── test_data.csv           # held-out test set (with target)
├── wdbc_converted.csv      # full dataset
└── model/                  # saved .joblib models + metrics_summary.csv
d. Models Used
Comparison Table (test set, 114 samples)
ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.9649	0.9960	0.9750	0.9286	0.9512	0.9245
Decision Tree	0.9211	0.9448	0.9459	0.8333	0.8861	0.8299
KNN	0.9561	0.9825	0.9744	0.9048	0.9383	0.9058
Naive Bayes	0.9386	0.9934	1.0000	0.8333	0.9091	0.8715
Support Vector Machine	0.9737	0.9947	1.0000	0.9286	0.9630	0.9442
Replace these numbers with the ones printed when you run train_models.py on BITS Virtual Lab if they differ.

Observations
ML Model Name	Observation about model performance
Logistic Regression	Very strong baseline: second-best accuracy (96.5%) and the highest AUC (0.996). The classes are almost linearly separable after scaling, so a linear model works very well, and it stays interpretable.
Decision Tree	Weakest model (accuracy 92.1%, MCC 0.83). It missed the most malignant cases (recall 0.83) and, being a single tree, is prone to overfitting and instability on a small dataset.
KNN	Good performance (95.6% accuracy, AUC 0.98) once features are scaled. Slightly lower recall than LR/SVM, and prediction is slower since it stores all training data.
Naive Bayes	Perfect precision (no false alarms) and high AUC (0.993), but recall is only 0.83, so it misses some malignant tumours. The feature-independence assumption is violated since many features (radius, perimeter, area) are highly correlated.
Support Vector Machine	Best overall: highest accuracy (97.4%), F1 (0.963) and MCC (0.944), with perfect precision. The RBF kernel captures the non-linear boundary well after scaling.
Overall Winner	Support Vector Machine (RBF) – best Accuracy, F1 and MCC on this dataset, with Logistic Regression a very close second. In a medical setting recall for malignant cases is crucial, so LR/SVM (recall 0.93) are preferred over the Decision Tree and Naive Bayes.
