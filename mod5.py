"""
MODULE 5 - Scam Type Classification

This stage is trained only on rows whose verdict is SCAM. At prediction time,
the saved verdict model runs first; scam type prediction runs only for SCAM.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "cleaned_india_scam_dataset.csv"
VERDICT_MODEL_FILE = BASE_DIR / "logistic_regression_model.joblib"
SCAM_TYPE_MODEL_FILE = BASE_DIR / "scam_type_model.joblib"
RESULTS_FILE = BASE_DIR / "scam_type_results.csv"
CONFUSION_MATRIX_FILE = BASE_DIR / "scam_type_confusion_matrix.csv"

TEXT_COL = "message"
VERDICT_COL = "verdict"
SCAM_TYPE_COL = "scam_type"

# The dataset also contains CHARITY_SCAM. It is retained because it is a real
# scam label rather than being silently removed from training.
SCAM_TYPE_ALIASES = {
    "RECRUITMENT_SCAM": "JOB_SCAM",
}


def normalize_verdict(value):
    return (
        str(value)
        .strip()
        .upper()
        .replace("LEGIT", "LEGITIMATE")
        .replace("MISINFORMATION", "FALSE_MISINFORMATION")
    )


def load_scam_training_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    required = {TEXT_COL, VERDICT_COL, SCAM_TYPE_COL}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str).str.strip()
    df[VERDICT_COL] = df[VERDICT_COL].fillna("").map(normalize_verdict)
    df[SCAM_TYPE_COL] = (
        df[SCAM_TYPE_COL]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(SCAM_TYPE_ALIASES)
    )

    scam_df = df[
        (df[VERDICT_COL] == "SCAM")
        & (df[TEXT_COL] != "")
        & (df[SCAM_TYPE_COL] != "")
        & (df[SCAM_TYPE_COL] != "NONE")
    ].copy()

    if scam_df.empty:
        raise ValueError("No usable SCAM rows with a scam_type were found.")

    return scam_df


def train_scam_type_model():
    df = load_scam_training_data()
    print(f"Training rows: {len(df)}")
    print("Scam types:")
    print(df[SCAM_TYPE_COL].value_counts().to_string())

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[SCAM_TYPE_COL])
    text_train, text_test, y_train, y_test = train_test_split(
        df[TEXT_COL],
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(text_train)
    x_test = vectorizer.transform(text_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nScam type accuracy: {accuracy:.4f}")
    print(
        classification_report(
            y_test,
            predictions,
            labels=range(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    results = pd.DataFrame(
        classification_report(
            y_test,
            predictions,
            labels=range(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    results.to_csv(RESULTS_FILE)

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
            "vectorizer": vectorizer,
            "label_encoder": label_encoder,
        },
        SCAM_TYPE_MODEL_FILE,
    )
    print(f"Saved -> {SCAM_TYPE_MODEL_FILE.name}")
    print(f"Saved -> {RESULTS_FILE.name}")
    print(f"Saved -> {CONFUSION_MATRIX_FILE.name}")


def predict_message(message):
    """Run verdict first, then classify scam type only when verdict is SCAM."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")
    if not VERDICT_MODEL_FILE.exists() or not SCAM_TYPE_MODEL_FILE.exists():
        raise FileNotFoundError("Run mod4.py and mod5.py before predicting messages.")

    verdict_bundle = joblib.load(VERDICT_MODEL_FILE)
    scam_type_bundle = joblib.load(SCAM_TYPE_MODEL_FILE)

    verdict_features = verdict_bundle["vectorizer"].transform([message])
    verdict_id = verdict_bundle["model"].predict(verdict_features)[0]
    verdict = verdict_bundle["label_encoder"].inverse_transform([verdict_id])[0]

    result = {"message": message, "verdict": verdict, "scam_type": None}
    if verdict == "SCAM":
        scam_features = scam_type_bundle["vectorizer"].transform([message])
        scam_id = scam_type_bundle["model"].predict(scam_features)[0]
        result["scam_type"] = scam_type_bundle["label_encoder"].inverse_transform([scam_id])[0]

    return result


if __name__ == "__main__":
    train_scam_type_model()