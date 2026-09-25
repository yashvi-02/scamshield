import os
import re
import pandas as pd
import numpy as np

DATASET_FILE = "india_scam_detection_master_5000.csv"
OUTPUT_FILE = "cleaned_india_scam_dataset_minimal.csv"
KEEP_COLUMNS = ["message", "verdict", "language"]


def normalize_text(value):
    if pd.isna(value):
        return ""

    text = str(value)
    text = text.replace("\u00a0", " ")
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.strip(" \t\"'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_verdict(value):
    if pd.isna(value):
        return "unknown"

    verdict = str(value).strip().upper()
    mapping = {
        "SCAM": "scam",
        "LEGITIMATE": "legit",
        "FALSE_MISINFORMATION": "misinformation",
        "SUSPICIOUS": "suspicious",
        "FAKE": "scam",
        "FRAUD": "scam",
        "SAFE": "legit",
        "NOT_SCAM": "legit",
        "NON_SCAM": "legit",
    }
    return mapping.get(verdict, verdict.lower())


def normalize_language(value):
    if pd.isna(value):
        return "unknown"
    lang = str(value).strip().upper()
    lang = lang.replace(" ", "_")
    lang = lang.replace("-", "_")
    return lang


def normalize_yes_no(value):
    if pd.isna(value):
        return 0
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return 1
        if value in {"0", "false", "no", "n"}:
            return 0
    return int(value)


def main():
    print(f"Loading dataset: {DATASET_FILE}")

    try:
        df = pd.read_csv(DATASET_FILE, encoding="utf-8-sig")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if "message" not in df.columns:
        raise ValueError("Dataset must contain a 'message' column for scam text detection.")

    if "verdict" not in df.columns:
        raise ValueError("Dataset must contain a 'verdict' column for class labels.")

    df = df.copy()
    df = df[[col for col in KEEP_COLUMNS if col in df.columns]].copy()
    df = df.dropna(subset=["message"]).copy()
    df = df.drop_duplicates(subset=["message", "verdict"], keep="first")

    df["message"] = df["message"].apply(normalize_text)
    df = df[df["message"].str.len() > 2].copy()

    if "verdict" in df.columns:
        df["verdict"] = df["verdict"].apply(normalize_verdict)
        df = df[df["verdict"] != "unknown"].copy()

    if "language" in df.columns:
        df["language"] = df["language"].apply(normalize_language)

    df = df.reset_index(drop=True)

    output_path = os.path.join(".", OUTPUT_FILE)
    df.to_csv(output_path, index=False)

    print(f"Cleaned shape: {df.shape}")
    print(f"Columns kept: {list(df.columns)}")
    print(f"Verdict distribution:\n{df['verdict'].value_counts().to_string()}")
    print(f"Language distribution:\n{df['language'].value_counts().to_string()}")
    print(f"Saved cleaned dataset to: {output_path}")


if __name__ == "__main__":
    main()
