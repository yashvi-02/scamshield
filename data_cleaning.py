import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_FILE="india_scam_detection_master_5000.csv"
OUTPUT_FILE="cleaned_india_scam_dataset.csv"

RANDOM_STATE=42

EXPECTED_COLUMNS=[
    "message","scam_type","risk_level","evidence","requested_action",
    "language","contains_url","contains_phone","asks_for_otp",
    "asks_for_upi_pin","asks_for_money","impersonates_authority","urgency",
    "verdict"
]

def normalize_text(value):
    if pd.isna(value):
        return ""
    text=str(value)
    text=text.replace("\u00a0"," ")
    text=text.replace("\r"," ").replace("\n"," ")
    text=re.sub(r"\s+"," ",text)
    return text.strip(" \t\"'")

def normalize_verdict(value):
    if pd.isna(value):
        return ""
    verdict=str(value).strip().upper()
    mapping={
        "SCAM":"SCAM",
        "LEGITIMATE":"LEGITIMATE",
        "LEGIT":"LEGITIMATE",
        "FALSE_MISINFORMATION":"FALSE_MISINFORMATION",
        "MISINFORMATION":"FALSE_MISINFORMATION",
        "SUSPICIOUS":"SUSPICIOUS",
        "FAKE":"SCAM",
        "FRAUD":"SCAM",
        "SAFE":"LEGITIMATE",
        "NOT_SCAM":"LEGITIMATE",
        "NON_SCAM":"LEGITIMATE"
    }
    return mapping.get(verdict,verdict)

def normalize_language(value):
    if pd.isna(value):
        return "UNKNOWN"
    return str(value).strip().upper().replace(" ","_").replace("-","_")

def normalize_binary(value):
    if pd.isna(value):
        return 0
    if isinstance(value,str):
        value=value.strip().lower()
        if value in {"1","true","yes","y"}:
            return 1
        if value in {"0","false","no","n"}:
            return 0
    try:
        return int(value)
    except:
        return 0

def normalize_risk(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()

def normalize_for_duplicate_check(text):
    text=text.lower()
    text=re.sub(r"\d+","<num>",text)
    text=re.sub(r"https?://\S+","<url>",text)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

def main():
    print("="*70)
    print("MODULE 1 - DATASET VALIDATION & PREPARATION")
    print("="*70)

    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")

    df=pd.read_csv(DATASET_FILE,encoding="utf-8-sig")

    print(f"\nOriginal shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")

    missing_columns=[c for c in EXPECTED_COLUMNS if c not in df.columns]

    if missing_columns:
        print("\nWARNING: Missing expected columns:")
        print(missing_columns)

    if "message" not in df.columns or "verdict" not in df.columns:
        raise ValueError("Dataset must contain message and verdict columns.")

    # -----------------------------
    # BASIC CLEANING
    # -----------------------------

    df=df.copy()

    df["message"]=df["message"].apply(normalize_text)
    df["verdict"]=df["verdict"].apply(normalize_verdict)

    if "language" in df.columns:
        df["language"]=df["language"].apply(normalize_language)

    for col in [
        "contains_url",
        "contains_phone",
        "asks_for_otp",
        "asks_for_upi_pin",
        "asks_for_money",
        "impersonates_authority"
    ]:
        if col in df.columns:
            df[col]=df[col].apply(normalize_binary)

    if "risk_level" in df.columns:
        df["risk_level"]=df["risk_level"].apply(normalize_risk)

    # Remove empty messages
    before=len(df)
    df=df[df["message"].str.len()>2].copy()
    print(f"\nRemoved empty/invalid messages: {before-len(df)}")

    # Remove unknown labels
    valid_labels={
        "SCAM",
        "LEGITIMATE",
        "SUSPICIOUS",
        "FALSE_MISINFORMATION"
    }

    before=len(df)
    df=df[df["verdict"].isin(valid_labels)].copy()
    print(f"Removed invalid verdict labels: {before-len(df)}")

    # -----------------------------
    # EXACT DUPLICATES
    # -----------------------------

    exact_duplicates=df.duplicated(subset=["message"],keep="first").sum()

    print(f"\nExact duplicate messages: {exact_duplicates}")

    df=df.drop_duplicates(
        subset=["message"],
        keep="first"
    ).copy()

    # -----------------------------
    # TEMPLATE DUPLICATES
    # -----------------------------

    df["template_key"]=df["message"].apply(normalize_for_duplicate_check)

    template_duplicates=df.duplicated(
        subset=["template_key"],
        keep="first"
    ).sum()

    print(f"Template-like duplicates: {template_duplicates}")

    df=df.drop_duplicates(
        subset=["template_key"],
        keep="first"
    ).copy()

    df=df.drop(columns=["template_key"])

    # -----------------------------
    # MISSING VALUE REPORT
    # -----------------------------

    print("\nMissing values:")
    print(df.isnull().sum().to_string())

    # Fill optional text columns
    for col in ["scam_type","risk_level","evidence","requested_action","language"]:
        if col in df.columns:
            df[col]=df[col].fillna("UNKNOWN").astype(str)

    # Fill binary columns
    for col in [
        "contains_url",
        "contains_phone",
        "asks_for_otp",
        "asks_for_upi_pin",
        "asks_for_money",
        "impersonates_authority"
    ]:
        if col in df.columns:
            df[col]=df[col].fillna(0).astype(int)

    df=df.reset_index(drop=True)

    # -----------------------------
    # CLASS DISTRIBUTION
    # -----------------------------

    print("\nFinal class distribution:")
    print(df["verdict"].value_counts().to_string())

    print("\nClass percentages:")
    print(
        (df["verdict"].value_counts(normalize=True)*100)
        .round(2)
        .to_string()
    )

    # -----------------------------
    # LANGUAGE DISTRIBUTION
    # -----------------------------

    if "language" in df.columns:
        print("\nLanguage distribution:")
        print(df["language"].value_counts().to_string())

        print("\nVerdict x Language:")
        print(
            pd.crosstab(
                df["verdict"],
                df["language"]
            ).to_string()
        )

    # -----------------------------
    # SCAM TYPE DISTRIBUTION
    # -----------------------------

    if "scam_type" in df.columns:
        print("\nScam type distribution:")
        print(
            df.loc[
                df["verdict"]=="SCAM",
                "scam_type"
            ].value_counts().to_string()
        )

    # -----------------------------
    # ANNOTATION VALIDATION
    # -----------------------------

    print("\n" + "="*70)
    print("ANNOTATION VALIDATION")
    print("="*70)

    if "risk_level" in df.columns:

        scam_low=df[
            (df["verdict"]=="SCAM") &
            (df["risk_level"]=="LOW")
        ]

        scam_medium=df[
            (df["verdict"]=="SCAM") &
            (df["risk_level"]=="MEDIUM")
        ]

        print(f"SCAM labelled LOW risk: {len(scam_low)}")
        print(f"SCAM labelled MEDIUM risk: {len(scam_medium)}")

    if "asks_for_otp" in df.columns:

        otp_messages=df["message"].str.contains(
            r"\botp\b|one[- ]time password|verification code",
            case=False,
            regex=True,
            na=False
        )

        contradictory_otp=df[
            otp_messages &
            (df["asks_for_otp"]==0)
        ]

        print(
            f"Possible OTP annotation contradictions: "
            f"{len(contradictory_otp)}"
        )

    if "asks_for_upi_pin" in df.columns:

        upi_pin_messages=df["message"].str.contains(
            r"\bupi\s*(pin|password)|upi pin",
            case=False,
            regex=True,
            na=False
        )

        contradictory_upi=df[
            upi_pin_messages &
            (df["asks_for_upi_pin"]==0)
        ]

        print(
            f"Possible UPI PIN annotation contradictions: "
            f"{len(contradictory_upi)}"
        )

    # -----------------------------
    # SAVE CLEAN DATASET
    # -----------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nSaved cleaned dataset: {OUTPUT_FILE}")
    print(f"Final cleaned shape: {df.shape}")

    # -----------------------------
    # STRATIFIED 70/15/15 SPLIT
    # -----------------------------

    print("\nCreating stratified train/validation/test split...")

    train_df,temp_df=train_test_split(
        df,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=df["verdict"]
    )

    validation_df,test_df=train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df["verdict"]
    )

    os.makedirs("data/processed",exist_ok=True)

    train_df.to_csv(
        "data/processed/train.csv",
        index=False,
        encoding="utf-8-sig"
    )

    validation_df.to_csv(
        "data/processed/validation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_df.to_csv(
        "data/processed/test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nSplit sizes:")
    print(f"Train:      {len(train_df)}")
    print(f"Validation: {len(validation_df)}")
    print(f"Test:       {len(test_df)}")

    print("\nTrain distribution:")
    print(train_df["verdict"].value_counts().to_string())

    print("\nValidation distribution:")
    print(validation_df["verdict"].value_counts().to_string())

    print("\nTest distribution:")
    print(test_df["verdict"].value_counts().to_string())

    print("\n" + "="*70)
    print("MODULE 1 COMPLETED")
    print("="*70)

if __name__=="__main__":
    main()