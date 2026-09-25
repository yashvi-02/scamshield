import re
import pandas as pd


INPUT_FILE = "cleaned_india_scam_dataset.csv"
OUTPUT_FILE = "nlp_preprocessed_dataset.csv"


def preprocess_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Replace URLs
    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' URLTOKEN ',
        text
    )

    # Replace phone numbers
    text = re.sub(
        r'\b(?:\+91[-\s]?)?[6-9]\d{9}\b',
        ' PHONETOKEN ',
        text
    )

    # Normalize OTP
    text = re.sub(
        r'\bo\.?t\.?p\.?\b',
        ' otp ',
        text
    )

    # Normalize UPI
    text = re.sub(
        r'\bu\.?p\.?i\.?\b',
        ' upi ',
        text
    )

    # Normalize KYC
    text = re.sub(
        r'\bk\.?y\.?c\.?\b',
        ' kyc ',
        text
    )

    # Remove extra spaces
    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


def main():

    print("Loading cleaned dataset...")

    df = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig"
    )

    print("Original shape:", df.shape)

    # Check message column
    if "message" not in df.columns:
        print("ERROR: message column not found.")
        return

    # Apply NLP preprocessing
    df["processed_message"] = df["message"].apply(
        preprocess_text
    )

    # Remove empty messages
    df = df[
        df["processed_message"].str.len() > 2
    ].copy()

    # Reset index
    df = df.reset_index(drop=True)

    # Save preprocessed dataset
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("NLP preprocessing completed.")
    print("Final shape:", df.shape)

    print()
    print("Sample results:")
    print("-" * 60)

    for i in range(min(5, len(df))):

        print("Original:")
        print(df.loc[i, "message"])

        print()

        print("Processed:")
        print(df.loc[i, "processed_message"])

        print("-" * 60)

    print()
    print("Saved as:", OUTPUT_FILE)


if __name__ == "__main__":
    main()