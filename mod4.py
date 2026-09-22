"""
MODULE 4 — ML Model Comparison
India Scam Detection — 4-class verdict prediction
(SCAM / LEGITIMATE / SUSPICIOUS / FALSE_MISINFORMATION)

Models:
  A. Logistic Regression   (TF-IDF)
  B. Linear SVM            (TF-IDF)
  C. Naive Bayes           (TF-IDF)
  D. Random Forest         (TF-IDF + structured features)

Run in VS Code:
  1. Open this folder in VS Code
  2. Create/activate a virtual env (recommended):
        python -m venv venv
        venv\\Scripts\\activate      (Windows)
        source venv/bin/activate     (Mac/Linux)
  3. Install requirements:
        pip install pandas numpy scikit-learn scipy matplotlib seaborn
  4. Place india_scam_detection_master_5000.csv in the same folder
  5. Run: python module4_model_comparison.py
"""

from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "cleaned_india_scam_dataset.csv"
TARGET_COL = "verdict"
TEXT_COL = "message"
STRUCTURED_COLS = [
    "contains_url", "contains_phone", "asks_for_otp",
    "asks_for_upi_pin", "asks_for_money", "impersonates_authority",
]

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

if TEXT_COL not in df.columns or TARGET_COL not in df.columns:
    raise ValueError(f"Dataset must contain '{TEXT_COL}' and '{TARGET_COL}' columns.")

df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
df[TARGET_COL] = (
    df[TARGET_COL]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
    .replace({
        "LEGIT": "LEGITIMATE",
        "FAKE": "SCAM",
        "FRAUD": "SCAM",
        "MISINFORMATION": "FALSE_MISINFORMATION",
    })
)
df = df[(df[TEXT_COL] != "") & (df[TARGET_COL] != "")].copy()

missing_structured = [col for col in STRUCTURED_COLS if col not in df.columns]
for col in missing_structured:
    df[col] = 0
df[STRUCTURED_COLS] = df[STRUCTURED_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"Loaded {len(df)} rows")
print(df["verdict"].value_counts(), "\n")

# ----------------------------------------------------------------------
# 2. ENCODE LABELS
# ----------------------------------------------------------------------
le = LabelEncoder()
y = le.fit_transform(df[TARGET_COL])
class_names = le.classes_
print("Classes:", list(class_names))

# Identify which encoded index is SCAM (needed for recall-on-SCAM reporting)
scam_idx = list(class_names).index("SCAM")

# ----------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT  (stratified so class balance is preserved)
# ----------------------------------------------------------------------
X_text = df[TEXT_COL]
X_struct = df[STRUCTURED_COLS].values

X_text_train, X_text_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
    X_text, X_struct, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# ----------------------------------------------------------------------
# 4. TF-IDF VECTORIZATION (fit ONLY on train, transform test)
# ----------------------------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),      # unigrams + bigrams catch phrases like "verify otp"
    stop_words="english",
    sublinear_tf=True
)

X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test = tfidf.transform(X_text_test)

# Combined TF-IDF + structured features (for Random Forest)
X_combined_train = hstack([X_tfidf_train, csr_matrix(X_struct_train)])
X_combined_test = hstack([X_tfidf_test, csr_matrix(X_struct_test)])

# ----------------------------------------------------------------------
# 5. HELPER: TRAIN + EVALUATE
# ----------------------------------------------------------------------
results = []
fitted_models = {}

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec_macro = precision_score(y_test, preds, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, preds, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)

    # SCAM-specific recall — the number that matters most for this project
    scam_recall = recall_score(y_test, preds, labels=[scam_idx], average="micro", zero_division=0)
    scam_precision = precision_score(y_test, preds, labels=[scam_idx], average="micro", zero_division=0)
    scam_f1 = f1_score(y_test, preds, labels=[scam_idx], average="micro", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision (macro)": round(prec_macro, 4),
        "Recall (macro)": round(rec_macro, 4),
        "F1 (macro)": round(f1_macro, 4),
        "SCAM Recall": round(scam_recall, 4),
        "SCAM Precision": round(scam_precision, 4),
        "SCAM F1": round(scam_f1, 4),
    })

    fitted_models[name] = (model, preds)

    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print(classification_report(y_test, preds, target_names=class_names, zero_division=0))

    return preds


# ----------------------------------------------------------------------
# 6. MODEL A — LOGISTIC REGRESSION (TF-IDF)
# ----------------------------------------------------------------------
logreg = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",   # important: protects recall on minority/critical class
    random_state=RANDOM_STATE
)
evaluate_model("Logistic Regression", logreg, X_tfidf_train, X_tfidf_test, y_train, y_test)

# ----------------------------------------------------------------------
# 7. MODEL B — LINEAR SVM (TF-IDF)
# ----------------------------------------------------------------------
svm = LinearSVC(
    class_weight="balanced",
    random_state=RANDOM_STATE,
    max_iter=5000
)
evaluate_model("Linear SVM", svm, X_tfidf_train, X_tfidf_test, y_train, y_test)

# ----------------------------------------------------------------------
# 8. MODEL C — NAIVE BAYES (TF-IDF)
# ----------------------------------------------------------------------
nb = MultinomialNB()
evaluate_model("Naive Bayes", nb, X_tfidf_train, X_tfidf_test, y_train, y_test)

# ----------------------------------------------------------------------
# 9. MODEL D — RANDOM FOREST (TF-IDF + structured features)
# ----------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
evaluate_model("Random Forest (TF-IDF + structured)", rf, X_combined_train, X_combined_test, y_train, y_test)

# ----------------------------------------------------------------------
# 10. COMPARISON TABLE
# ----------------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n\n" + "=" * 90)
print("MODEL COMPARISON TABLE")
print("=" * 90)
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison_results.csv", index=False)
print("\nSaved -> model_comparison_results.csv")

# Rank by SCAM Recall (the metric that matters most per project brief)
best_by_scam_recall = results_df.sort_values("SCAM Recall", ascending=False).iloc[0]
print(f"\nBest model by SCAM recall: {best_by_scam_recall['Model']} "
      f"(SCAM Recall = {best_by_scam_recall['SCAM Recall']})")

# ----------------------------------------------------------------------
# 11. CONFUSION MATRICES (visual)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for ax, (name, (model, preds)) in zip(axes, fitted_models.items()):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrices.png", dpi=150)
print("Saved -> confusion_matrices.png")

# Save the fitted logistic-regression model and vectorizer for later predictions.
joblib.dump(
    {"model": logreg, "vectorizer": tfidf, "label_encoder": le},
    BASE_DIR / "logistic_regression_model.joblib",
)
print("Saved -> logistic_regression_model.joblib")
