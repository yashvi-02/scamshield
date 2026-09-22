"""
MODULE 6 - Explainable Risk Engine

Combines the verdict model's scam probability with message indicators. The
result contains a risk level, score, detected signals, and human-readable
reasons instead of returning only a black-box prediction.
"""

from pathlib import Path
import re

import joblib


BASE_DIR = Path(__file__).resolve().parent
VERDICT_MODEL_FILE = BASE_DIR / "logistic_regression_model.joblib"
SCAM_TYPE_MODEL_FILE = BASE_DIR / "scam_type_model.joblib"

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9-]+\.(?:com|in|net|org|co)\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"(?:\+?91[-\s]?)?[6-9]\d{9}\b")
OTP_PATTERN = re.compile(r"\b(?:otp|one[- ]time password|verification code)\b", re.IGNORECASE)
UPI_PATTERN = re.compile(r"\b(?:upi|upi pin|vpa|gpay|google pay|phonepe|paytm)\b", re.IGNORECASE)
MONEY_PATTERN = re.compile(
    r"\b(?:pay|payment|paid|send|transfer|deposit|recharge|donate|refund|money|fee|amount|rs\.?|inr)\b|₹",
    re.IGNORECASE,
)
AUTHORITY_PATTERN = re.compile(
    r"\b(?:police|court|government|income tax|tax department|bank|rbi|electricity department|customs|officer|authority)\b",
    re.IGNORECASE,
)
URGENCY_PATTERN = re.compile(
    r"\b(?:urgent|immediately|now|today|within \d+ hours?|last chance|blocked|suspended|expire|expires|deadline)\b",
    re.IGNORECASE,
)
CREDENTIAL_PATTERN = re.compile(
    r"\b(?:password|passcode|pin|cvv|card number|account number|login|credentials|kyc)\b",
    re.IGNORECASE,
)
THREAT_PATTERN = re.compile(
    r"\b(?:arrest|legal action|penalty|fine|police case|disconnect|block(?:ed)?|suspend(?:ed)?)\b",
    re.IGNORECASE,
)

INDICATOR_WEIGHTS = {
    "suspicious_url": 10,
    "phone_number": 3,
    "otp_request": 12,
    "upi_or_wallet": 12,
    "money_request": 10,
    "authority_impersonation": 10,
    "urgent_language": 8,
    "credential_request": 10,
    "threat_or_consequence": 8,
}


def _load_bundle(path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path.name}. Run mod4.py first.")
    return joblib.load(path)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def detect_indicators(message, metadata=None):
    """Return boolean risk indicators from text and optional dataset metadata."""
    metadata = metadata or {}
    indicators = {
        "suspicious_url": bool(URL_PATTERN.search(message)),
        "phone_number": bool(PHONE_PATTERN.search(message)),
        "otp_request": bool(OTP_PATTERN.search(message)),
        "upi_or_wallet": bool(UPI_PATTERN.search(message)),
        "money_request": bool(MONEY_PATTERN.search(message)),
        "authority_impersonation": bool(AUTHORITY_PATTERN.search(message)),
        "urgent_language": bool(URGENCY_PATTERN.search(message)),
        "credential_request": bool(CREDENTIAL_PATTERN.search(message)),
        "threat_or_consequence": bool(THREAT_PATTERN.search(message)),
    }

    metadata_map = {
        "contains_url": "suspicious_url",
        "contains_phone": "phone_number",
        "asks_for_otp": "otp_request",
        "asks_for_upi_pin": "upi_or_wallet",
        "asks_for_money": "money_request",
        "impersonates_authority": "authority_impersonation",
    }
    for source_key, indicator_key in metadata_map.items():
        if source_key in metadata:
            indicators[indicator_key] = indicators[indicator_key] or _as_bool(metadata[source_key])

    return indicators


def _risk_level(score):
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


def _recommended_action(level, verdict):
    if level == "CRITICAL":
        return "Do not click, pay, share credentials, or reply. Verify through an official channel."
    if level == "HIGH":
        return "Pause and independently verify the sender before taking any action."
    if level == "MEDIUM":
        return "Treat cautiously and verify the request using a trusted source."
    if verdict == "LEGITIMATE":
        return "No strong scam signals detected, but remain alert to unexpected changes."
    return "No immediate high-risk combination detected."


def analyze_message(message, metadata=None):
    """Return verdict, scam type, risk score, signals, and explanations."""
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")

    verdict_bundle = _load_bundle(VERDICT_MODEL_FILE)
    scam_type_bundle = _load_bundle(SCAM_TYPE_MODEL_FILE)
    clean_message = message.strip()
    features = verdict_bundle["vectorizer"].transform([clean_message])
    verdict_model = verdict_bundle["model"]
    probabilities = verdict_model.predict_proba(features)[0]
    verdict_id = verdict_model.predict(features)[0]
    verdict = verdict_bundle["label_encoder"].inverse_transform([verdict_id])[0]
    class_names = verdict_bundle["label_encoder"].classes_
    probability_by_class = dict(zip(class_names, probabilities))
    scam_probability = float(probability_by_class.get("SCAM", 0.0))

    indicators = detect_indicators(clean_message, metadata)
    indicator_score = sum(
        INDICATOR_WEIGHTS[name] for name, detected in indicators.items() if detected
    )
    score = round(min(100.0, scam_probability * 60 + indicator_score), 2)

    # A non-SCAM model verdict must not be escalated to CRITICAL by a single
    # text rule, but the engine can still flag MEDIUM risk for review.
    if verdict != "SCAM":
        score = min(score, 49.0)

    level = _risk_level(score)
    reasons = [
        name.replace("_", " ").title()
        for name, detected in indicators.items()
        if detected
    ]
    if verdict == "SCAM":
        reasons.insert(0, f"ML model estimates {scam_probability:.0%} probability of SCAM")
    elif scam_probability > 0.25:
        reasons.insert(0, f"ML model estimates {scam_probability:.0%} probability of SCAM")

    scam_type = None
    if verdict == "SCAM":
        scam_features = scam_type_bundle["vectorizer"].transform([clean_message])
        scam_id = scam_type_bundle["model"].predict(scam_features)[0]
        scam_type = scam_type_bundle["label_encoder"].inverse_transform([scam_id])[0]

    return {
        "message": clean_message,
        "verdict": verdict,
        "scam_probability": round(scam_probability, 4),
        "scam_type": scam_type,
        "risk_score": score,
        "risk_level": level,
        "indicators": indicators,
        "indicator_score": indicator_score,
        "reasons": reasons,
        "recommended_action": _recommended_action(level, verdict),
    }


if __name__ == "__main__":
    example = (
        "Your bank account will be blocked today. Share your OTP and UPI PIN "
        "immediately at https://verify-account.example.com to avoid arrest."
    )
    print(analyze_message(example))