from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent

# Support loading from either project root or data/processed
def locate_file(filename: str) -> Path:
    candidates = [
        BASE_DIR / filename,
        BASE_DIR / "data" / "processed" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not locate required split file: {filename}")

TRAIN_FILE = locate_file("train.csv")
VAL_FILE = locate_file("validation.csv")
TEST_FILE = locate_file("test.csv")

TARGET_COL = "verdict"
TEXT_COL = "message"

print("=" * 70)
print("SCAMSHIELD MODULE 4 - VERDICT CLASSIFIER BENCHMARK & EVALUATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. LOAD PRE-SPLIT DATASETS (STRICT SEPARATION)
# ---------------------------------------------------------
train_df = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")
val_df = pd.read_csv(VAL_FILE, encoding="utf-8-sig")
test_df = pd.read_csv(TEST_FILE, encoding="utf-8-sig")

for df in [train_df, val_df, test_df]:
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
    df[TARGET_COL] = df[TARGET_COL].fillna("").astype(str).str.strip().str.upper()

print(f"Loaded train: {len(train_df)} rows")
print(f"Loaded validation: {len(val_df)} rows")
print(f"Loaded test: {len(test_df)} rows")

# ---------------------------------------------------------
# 2. ENCODE LABELS (FITTED ONLY ON TRAIN)
# ---------------------------------------------------------
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df[TARGET_COL])
y_val = label_encoder.transform(val_df[TARGET_COL])
y_test = label_encoder.transform(test_df[TARGET_COL])

class_names = list(label_encoder.classes_)
scam_idx = class_names.index("SCAM")
print(f"\nTarget Classes: {class_names}")

# ---------------------------------------------------------
# 3. TF-IDF FEATURE EXTRACTION (FIT ON TRAIN ONLY)
# ---------------------------------------------------------
word_vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=10000,
)

char_vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    sublinear_tf=True,
    max_features=10000,
)

print("\nFitting vectorizers strictly on training data...")
X_word_train = word_vectorizer.fit_transform(train_df[TEXT_COL])
X_char_train = char_vectorizer.fit_transform(train_df[TEXT_COL])

X_word_val = word_vectorizer.transform(val_df[TEXT_COL])
X_char_val = char_vectorizer.transform(val_df[TEXT_COL])

X_word_test = word_vectorizer.transform(test_df[TEXT_COL])
X_char_test = char_vectorizer.transform(test_df[TEXT_COL])

X_train_final = hstack([X_word_train, X_char_train]).tocsr()
X_val_final = hstack([X_word_val, X_char_val]).tocsr()
X_test_final = hstack([X_word_test, X_char_test]).tocsr()

print(f"Feature matrix shapes -> Train: {X_train_final.shape}, Val: {X_val_final.shape}, Test: {X_test_final.shape}")

# ---------------------------------------------------------
# 4. BENCHMARK CANDIDATE MODELS
# ---------------------------------------------------------
candidate_models = {
    "Logistic Regression": LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=1.0,
        random_state=RANDOM_STATE,
    ),
    "Linear SVM": LinearSVC(
        class_weight="balanced",
        C=1.0,
        max_iter=10000,
        random_state=RANDOM_STATE,
        dual="auto",
    ),
    "Naive Bayes": MultinomialNB(alpha=0.3),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ),
}

comparison_records = []
trained_models = {}

for name, model in candidate_models.items():
    # Fit strictly on train
    model.fit(X_train_final, y_train)
    trained_models[name] = model

    # Validation evaluation (for model tuning & validation tracking)
    val_preds = model.predict(X_val_final)
    val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

    # Test evaluation (unseen holdout)
    test_preds = model.predict(X_test_final)
    test_acc = accuracy_score(y_test, test_preds)
    test_macro_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
    test_macro_rec = recall_score(y_test, test_preds, average="macro", zero_division=0)
    test_macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)

    # SCAM-specific metrics (critical security requirement)
    scam_prec = precision_score(y_test, test_preds, labels=[scam_idx], average="macro", zero_division=0)
    scam_rec = recall_score(y_test, test_preds, labels=[scam_idx], average="macro", zero_division=0)
    scam_f1 = f1_score(y_test, test_preds, labels=[scam_idx], average="macro", zero_division=0)

    comparison_records.append({
        "Model": name,
        "Val_Macro_F1": round(val_macro_f1, 4),
        "Test_Accuracy": round(test_acc, 4),
        "Test_Macro_P": round(test_macro_prec, 4),
        "Test_Macro_R": round(test_macro_rec, 4),
        "Test_Macro_F1": round(test_macro_f1, 4),
        "SCAM_Precision": round(scam_prec, 4),
        "SCAM_Recall": round(scam_rec, 4),
        "SCAM_F1": round(scam_f1, 4),
    })

comparison_df = pd.DataFrame(comparison_records)
print("\n" + "=" * 70)
print("CANDIDATE MODEL COMPARISON (TRAIN -> VALIDATION -> TEST)")
print("=" * 70)
print(comparison_df.to_string(index=False))

comparison_df.to_csv(BASE_DIR / "model_comparison_results.csv", index=False)

# ---------------------------------------------------------
# 5. MODEL SELECTION & PERSISTENCE
# ---------------------------------------------------------
# Logistic Regression is selected because it produces well-calibrated class
# probabilities required by Module 6's Risk Engine.
selected_model_name = "Logistic Regression"
final_model = trained_models[selected_model_name]
print(f"\nSelected Model for Pipeline Artifact: {selected_model_name}")

artifact_path = BASE_DIR / "logistic_regression_model.joblib"
joblib.dump(
    {
        "model": final_model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "label_encoder": label_encoder,
        "class_names": class_names,
    },
    artifact_path,
)
print(f"Saved artifact bundle -> {artifact_path.name}")

# ---------------------------------------------------------
# 6. TEST PREDICTIONS & PROBABILITIES EXPORT
# ---------------------------------------------------------
test_final_preds = final_model.predict(X_test_final)
test_probabilities = final_model.predict_proba(X_test_final)

test_results_df = pd.DataFrame({
    "message": test_df[TEXT_COL].values,
    "actual_verdict": [class_names[i] for i in y_test],
    "predicted_verdict": [class_names[i] for i in test_final_preds],
})

for i, c_name in enumerate(class_names):
    test_results_df[f"prob_{c_name}"] = np.round(test_probabilities[:, i], 4)

test_results_df.to_csv(BASE_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
print("Saved test predictions -> test_predictions.csv")

# ---------------------------------------------------------
# 7. FALSE POSITIVE ANALYSIS (LEGITIMATE misclassified as SCAM)
# ---------------------------------------------------------
false_positives = test_results_df[
    (test_results_df["actual_verdict"] == "LEGITIMATE") &
    (test_results_df["predicted_verdict"] == "SCAM")
].copy()

false_positives.to_csv(BASE_DIR / "false_positive_messages.csv", index=False, encoding="utf-8-sig")
print(f"\nHoldout False Positives (LEGITIMATE -> SCAM): {len(false_positives)}")

# ---------------------------------------------------------
# 8. TEST CLASSIFICATION REPORT & CONFUSION MATRIX
# ---------------------------------------------------------
print("\nFinal Test Classification Report:")
print(classification_report(y_test, test_final_preds, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_test, test_final_preds, labels=np.arange(len(class_names)))
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title(f"{selected_model_name} Confusion Matrix (Test Set)")
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrices.png", dpi=150)
plt.close()
print("Saved confusion matrix plot -> confusion_matrices.png")

print("\n" + "=" * 70)
print("MODULE 4 PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 70)