"""
MODULE 5 - Scam Type Classification

Trained strictly on SCAM rows from train.csv.
Evaluated on SCAM rows from test.csv.
At inference, scam type prediction is invoked only when verdict == SCAM.
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent

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
TEST_FILE = locate_file("test.csv")
VERDICT_MODEL_FILE = BASE_DIR / "logistic_regression_model.joblib"
SCAM_TYPE_MODEL_FILE = BASE_DIR / "scam_type_model.joblib"
RESULTS_FILE = BASE_DIR / "scam_type_results.csv"
CONFUSION_MATRIX_FILE = BASE_DIR / "scam_type_confusion_matrix.csv"

TEXT_COL = "message"
VERDICT_COL = "verdict"
SCAM_TYPE_COL = "scam_type"

SCAM_TYPE_ALIASES = {
    "RECRUITMENT_SCAM": "JOB_SCAM",
}

def clean_scam_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
    df[VERDICT_COL] = df[VERDICT_COL].fillna("").astype(str).str.strip().str.upper()
    df[SCAM_TYPE_COL] = (
        df[SCAM_TYPE_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(SCAM_TYPE_ALIASES)
    )
    return df[
        (df[VERDICT_COL] == "SCAM")
        & (df[TEXT_COL] != "")
        & (df[SCAM_TYPE_COL] != "")
        & (df[SCAM_TYPE_COL] != "NONE")
        & (df[SCAM_TYPE_COL] != "UNKNOWN")
    ].copy()

def train_scam_type_model():
    print("=" * 70)
    print("SCAMSHIELD MODULE 5 - SCAM TYPE CLASSIFICATION")
    print("=" * 70)

    train_raw = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")
    test_raw = pd.read_csv(TEST_FILE, encoding="utf-8-sig")

    train_df = clean_scam_data(train_raw)
    test_df = clean_scam_data(test_raw)

    print(f"Training SCAM samples: {len(train_df)}")
    print(f"Holdout Test SCAM samples: {len(test_df)}")

    # Encode labels using classes found in training set
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[SCAM_TYPE_COL])
    
    # Filter test samples to known classes
    known_classes = set(label_encoder.classes_)
    test_df = test_df[test_df[SCAM_TYPE_COL].isin(known_classes)].copy()
    y_test = label_encoder.transform(test_df[SCAM_TYPE_COL])

    print(f"\nUnique Scam Subtypes ({len(label_encoder.classes_)}):")
    print(list(label_encoder.classes_))

    # Vectorizers fitted only on training SCAM messages
    word_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=5000,
        sublinear_tf=True,
    )

    X_w_train = word_vectorizer.fit_transform(train_df[TEXT_COL])
    X_c_train = char_vectorizer.fit_transform(train_df[TEXT_COL])
    X_train = hstack([X_w_train, X_c_train]).tocsr()

    X_w_test = word_vectorizer.transform(test_df[TEXT_COL])
    X_c_test = char_vectorizer.transform(test_df[TEXT_COL])
    X_test = hstack([X_w_test, X_c_test]).tocsr()

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        C=1.0,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nScam Type Holdout Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=range(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    # Save artifacts
    results_df = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            labels=range(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    results_df.to_csv(RESULTS_FILE)

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=range(len(label_encoder.classes_)),
    )
    pd.DataFrame(
        matrix,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    ).to_csv(CONFUSION_MATRIX_FILE)

    joblib.dump(
        {
            "model": model,
            "word_vectorizer": word_vectorizer,
            "char_vectorizer": char_vectorizer,
            "label_encoder": label_encoder,
            "classes": list(label_encoder.classes_),
        },
        SCAM_TYPE_MODEL_FILE,
    )
    print(f"Saved -> {SCAM_TYPE_MODEL_FILE.name}")
    print(f"Saved -> {RESULTS_FILE.name}")
    print(f"Saved -> {CONFUSION_MATRIX_FILE.name}")
    print("\n" + "=" * 70)
    print("MODULE 5 COMPLETED SUCCESSFULLY")
    print("=" * 70)

def predict_message(message: str):
    """Pipeline helper: Verdict classification first, then scam type only if SCAM."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if not VERDICT_MODEL_FILE.exists() or not SCAM_TYPE_MODEL_FILE.exists():
        raise FileNotFoundError("Both mod4 and mod5 artifacts must be built.")

    verdict_bundle = joblib.load(VERDICT_MODEL_FILE)
    scam_type_bundle = joblib.load(SCAM_TYPE_MODEL_FILE)

    clean_msg = message.strip()

    # 1. Verdict check
    v_w = verdict_bundle["word_vectorizer"].transform([clean_msg])
    v_c = verdict_bundle["char_vectorizer"].transform([clean_msg])
    v_feat = hstack([v_w, v_c]).tocsr()
    v_id = verdict_bundle["model"].predict(v_feat)[0]
    verdict = verdict_bundle["label_encoder"].inverse_transform([v_id])[0]

    result = {"message": clean_msg, "verdict": verdict, "scam_type": None}

    # 2. Scam type prediction ONLY for SCAM verdict
    if verdict == "SCAM":
        s_w = scam_type_bundle["word_vectorizer"].transform([clean_msg])
        s_c = scam_type_bundle["char_vectorizer"].transform([clean_msg])
        s_feat = hstack([s_w, s_c]).tocsr()
        s_id = scam_type_bundle["model"].predict(s_feat)[0]
        result["scam_type"] = scam_type_bundle["label_encoder"].inverse_transform([s_id])[0]

    return result

if __name__ == "__main__":
    train_scam_type_model()