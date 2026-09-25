from pathlib import Path
import re
import joblib
from scipy.sparse import hstack

BASE_DIR = Path(__file__).resolve().parent
VERDICT_MODEL_FILE = BASE_DIR / "logistic_regression_model.joblib"
SCAM_TYPE_MODEL_FILE = BASE_DIR / "scam_type_model.joblib"

# ---------------------------------------------------------------------------
# TRUSTED DOMAIN ALLOWLIST
# ---------------------------------------------------------------------------
TRUSTED_DOMAINS = {
    # Banking & Financial Institutions
    "onlinesbi.sbi", "sbi.co.in", "onlinesbi.com", "hdfcbank.com", "icicibank.com",
    "axisbank.com", "pnbindia.in", "bankofbaroda.in", "canarabank.com", "rbi.org.in",
    "npci.org.in",
    # Government & Public Utilities
    "gov.in", "nic.in", "uidai.gov.in", "incometax.gov.in", "indiapost.gov.in",
    "epfindia.gov.in", "digilocker.gov.in",
    # Legitimate Tech & Payment Platforms
    "google.com", "paytm.com", "phonepe.com", "amazon.in", "flipkart.com", "whatsapp.com",
}

# ---------------------------------------------------------------------------
# BILINGUAL HEURISTIC PATTERNS (English + Hinglish + Roman Hindi)
# ---------------------------------------------------------------------------
URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+|\b[a-z0-9-]+\.(?:com|in|net|org|co|xyz|top|ru|apk|site|online|live|club|vip)\b",
    re.I,
)
PHONE_PATTERN = re.compile(r"(?:\+?91[-\s]?)?[6-9]\d{9}\b")

# Direct OTP Harvesters
OTP_REQUEST_PATTERN = re.compile(
    r"\b(?:shar(?:e|ing|ed)?|send(?:ing)?|provid(?:e|ing|ed)?|enter(?:ing)?|giv(?:e|ing)?|tell(?:ing)?|submit(?:ting|ted)?|forward(?:ing)?|bhej(?:o|na|iye)?|de(?:na|do|iye)?|bataye?|dalein?)\b.{0,50}\b(?:otp|one[- ]time password|verification code)\b|"
    r"\b(?:otp|one[- ]time password|verification code)\b.{0,50}\b(?:shar(?:e|ing|ed)?|send(?:ing)?|provid(?:e|ing)?|enter(?:ing)?|giv(?:e|ing)?|tell(?:ing)?|submit(?:ting)?|forward(?:ing)?|bhej(?:o|na|iye)?|de(?:na|do|iye)?|bataye?|dalein?)\b",
    re.I,
)

# UPI PIN Theft (Explicitly asks to enter/give PIN to claim/receive funds)
UPI_PIN_THEFT_PATTERN = re.compile(
    r"\b(?:upi\s*pin|enter\s*pin|pin\s*dalein?|pin\s*share|share\s*pin|pin\s*enter|claim.*pin|cashback.*pin)\b",
    re.I,
)

# Casual Payment App Mention (Soft indicator, not sufficient alone)
UPI_APP_PATTERN = re.compile(
    r"\b(?:gpay|google pay|phonepe|paytm|vpa|bhim|upi)\b",
    re.I,
)

# Payment Demands & Monetary Terms
MONEY_PATTERN = re.compile(
    r"\b(?:pay|payment|transfer|deposit|recharge|donate|refund fee|processing fee|charges|paise|rupaye|bhejo|jama)\b|₹|\b(?:rs\.?|inr)\s*\d+",
    re.I,
)

# Authority Impersonation
AUTHORITY_PATTERN = re.compile(
    r"\b(?:police|court|government|income tax|tax department|bank|sbi|hdfc|icici|pnb|rbi|electricity|bijli|power corporation|customs|cbi|cyber cell|telecom|officer|adhikari)\b",
    re.I,
)

# High-Pressure Urgency
URGENCY_PATTERN = re.compile(
    r"\b(?:urgent(?:ly)?|immediately|act now|last chance|within \d+ (?:hours?|mins?|minutes?)|expires? today|deadline|respond immediately|action required|turant|jaldi|aaj hi|aaj raat|khatam ho jayega|turant call|sampark karein)\b",
    re.I,
)

# Sensitive Credential Requests
CREDENTIAL_PATTERN = re.compile(
    r"\b(?:password|passcode|cvv|card number|account number|login credentials|kyc|aadhaar|pan card|documents)\b",
    re.I,
)

# Threats of Disconnection, Arrest, or Penalties
THREAT_PATTERN = re.compile(
    r"\b(?:arrest(?:ed)?|legal action|penalty|fine|police case|disconnect(?:ed)?|block(?:ed)?|suspend(?:ed)?|deactivate|kat diya|kat jayega|band kar|band ho jayega|karvayi|action liya jayega)\b",
    re.I,
)

# Legitimate Conversational & P2P Context Overrides
CONVERSATIONAL_PATTERNS = [
    re.compile(r"\b(?:do not|don't|never|avoid|please don't|never share|do not share|mat karna|kisi ko na bataye)\b.{0,50}\b(?:otp|upi|pin|password|cvv|link|money)\b", re.I),
    re.compile(r"\b(?:otp|upi|pin|password|cvv)\b.{0,50}\b(?:do not|don't|never|avoid|mat karna)\b", re.I),
    re.compile(r"\b(?:split|party|dinner|lunch|chai|coffee|dost|bhai|yaar|kal ka|bill split|hisab|ticket)\b", re.I),
    re.compile(r"\b(?:happy birthday|good morning|good night|thank you|thanks|how are you|see you|take care|reach home|college|timetable|notes|assignment)\b", re.I),
]

INDICATOR_WEIGHTS = {
    "suspicious_url": 15,
    "phone_number": 4,
    "otp_request": 20,
    "upi_pin_theft": 20,
    "upi_app_mention": 6,
    "money_request": 8,
    "authority_impersonation": 15,
    "urgent_language": 12,
    "credential_request": 14,
    "threat_or_consequence": 16,
}


def _load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path.name}. Run mod4.py/mod5.py first.")
    return joblib.load(path)


def _bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def is_suspicious_url_found(text: str) -> bool:
    """Returns True ONLY if an external URL does NOT belong to our trusted domains."""
    found_urls = URL_PATTERN.findall(text)
    if not found_urls:
        return False
    for raw_u in found_urls:
        clean = (
            raw_u.lower()
            .replace("https://", "")
            .replace("http://", "")
            .replace("www.", "")
            .split("/")[0]
            .split("?")[0]
        )
        is_trusted = any(clean == d or clean.endswith("." + d) for d in TRUSTED_DOMAINS)
        if not is_trusted:
            return True
    return False


def detect_indicators(message: str, metadata: dict = None) -> dict:
    metadata = metadata or {}

    indicators = {
        "suspicious_url": is_suspicious_url_found(message),
        "phone_number": bool(PHONE_PATTERN.search(message)),
        "otp_request": bool(OTP_REQUEST_PATTERN.search(message)),
        "upi_pin_theft": bool(UPI_PIN_THEFT_PATTERN.search(message)),
        "upi_app_mention": bool(UPI_APP_PATTERN.search(message)),
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
        "asks_for_upi_pin": "upi_pin_theft",
        "asks_for_money": "money_request",
        "impersonates_authority": "authority_impersonation",
    }

    for source, target in metadata_map.items():
        if source in metadata:
            indicators[target] = indicators[target] or _bool(metadata[source])

    return indicators


def is_conversational_context(message: str) -> bool:
    return any(p.search(message) for p in CONVERSATIONAL_PATTERNS)


def risk_level(score: float) -> str:
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


def recommended_action(level: str, verdict: str) -> str:
    if verdict == "LEGITIMATE":
        return "No scam indicators detected. Standard caution is sufficient."
    if verdict == "FALSE_MISINFORMATION":
        return "Do not forward. Verify claim through official fact-checks or trusted news sources."
    if verdict == "SUSPICIOUS":
        return "Do not click unverified links or share personal details."
    if level == "CRITICAL":
        return "Do not click, pay, or share credentials/OTP. Contact the official institution directly."
    if level == "HIGH":
        return "Pause and independently verify the sender through official applications."
    return "Treat cautiously and verify using trusted sources."


def analyze_message(message: str, metadata: dict = None) -> dict:
    if not isinstance(message, str) or not message.strip():
        raise ValueError("message must be a non-empty string")

    verdict_bundle = _load(VERDICT_MODEL_FILE)
    scam_type_bundle = _load(SCAM_TYPE_MODEL_FILE)

    clean_message = message.strip()

    word_features = verdict_bundle["word_vectorizer"].transform([clean_message])
    char_features = verdict_bundle["char_vectorizer"].transform([clean_message])
    features = hstack([word_features, char_features]).tocsr()

    model = verdict_bundle["model"]
    probabilities = model.predict_proba(features)[0]
    class_names = verdict_bundle["label_encoder"].classes_
    prob_dict = dict(zip(class_names, probabilities))

    scam_probability = float(prob_dict.get("SCAM", 0.0))
    legit_probability = float(prob_dict.get("LEGITIMATE", 0.0))
    suspicious_probability = float(prob_dict.get("SUSPICIOUS", 0.0))
    misinformation_probability = float(prob_dict.get("FALSE_MISINFORMATION", 0.0))

    predicted_id = int(model.predict(features)[0])
    model_verdict = verdict_bundle["label_encoder"].inverse_transform([predicted_id])[0]

    indicators = detect_indicators(clean_message, metadata)
    indicator_score = sum(INDICATOR_WEIGHTS[k] for k, v in indicators.items() if v)

    # Hard scam signals that directly imply malice
    hard_scam_signals = [
        indicators["otp_request"],
        indicators["upi_pin_theft"],
        indicators["credential_request"],
        indicators["suspicious_url"],
        (indicators["authority_impersonation"] and indicators["threat_or_consequence"]),
        (indicators["threat_or_consequence"] and indicators["urgent_language"]),
    ]
    has_hard_signal = any(hard_scam_signals)
    is_conv = is_conversational_context(clean_message)

    # -----------------------------------------------------------------------
    # BALANCED ARBITRATION RULES
    # -----------------------------------------------------------------------
    # 1. P2P casual chats / Bill splitting without hard malicious signals
    if is_conv and not has_hard_signal:
        verdict = "LEGITIMATE"

    # 2. Hard Scam Attack Signals present (OTP, PIN, credential theft, threat + urgency, or bad link)
    elif has_hard_signal and (scam_probability >= 0.35 or indicator_score >= 20):
        verdict = "SCAM"

    # 3. Extremely high ML scam confidence on non-conversational text without trusted context
    elif model_verdict == "SCAM" and scam_probability >= 0.85 and not is_conv and not (URL_PATTERN.search(clean_message) and not indicators["suspicious_url"]):
        verdict = "SCAM"

    # 4. Misinformation handling
    elif model_verdict == "FALSE_MISINFORMATION" and misinformation_probability >= 0.45:
        verdict = "FALSE_MISINFORMATION"

    # 5. Informational notifications with ONLY trusted official links and NO threats/credential requests
    elif not has_hard_signal and not indicators["threat_or_consequence"] and not indicators["urgent_language"] and not indicators["credential_request"]:
        verdict = "LEGITIMATE"

    # 6. Unverified/unknown links or combined soft indicators
    elif indicators["suspicious_url"] or indicator_score >= 20 or scam_probability >= 0.55:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LEGITIMATE"

    # Calculate deterministic risk score (0-100)
    if verdict == "SCAM":
        score = min(100.0, max(55.0, scam_probability * 40.0 + indicator_score * 0.8))
    elif verdict == "SUSPICIOUS":
        score = min(49.0, max(25.0, scam_probability * 30.0 + indicator_score * 0.4))
    else:
        score = min(15.0, indicator_score * 0.2 + 2.0)

    score = round(score, 2)
    level = risk_level(score)

    if verdict == "LEGITIMATE":
        level = "LOW"
    elif verdict == "FALSE_MISINFORMATION":
        level = "MEDIUM" if misinformation_probability >= 0.5 else "LOW"

    # Construct clean reasons list
    reasons = [
        name.replace("_", " ").title()
        for name, detected in indicators.items()
        if detected and name != "upi_app_mention"
    ]

    if verdict == "SCAM":
        reasons.insert(0, f"ML model estimates {scam_probability:.0%} scam likelihood")
    elif verdict == "SUSPICIOUS":
        reasons.insert(0, f"ML model estimates {scam_probability:.0%} scam likelihood")
    elif verdict == "LEGITIMATE":
        reasons = ["No strong scam indicators were detected"]
    else:
        reasons.insert(0, f"ML model estimates {misinformation_probability:.0%} misinformation probability")

    # Predict scam subtype ONLY when final verdict is SCAM
    scam_type = None
    if verdict == "SCAM":
        s_w = scam_type_bundle["word_vectorizer"].transform([clean_message])
        s_c = scam_type_bundle["char_vectorizer"].transform([clean_message])
        s_feat = hstack([s_w, s_c]).tocsr()
        scam_id = scam_type_bundle["model"].predict(s_feat)[0]
        scam_type = scam_type_bundle["label_encoder"].inverse_transform([scam_id])[0]

    # Confidence calibration
    if verdict == "SCAM":
        confidence = max(scam_probability, 0.80)
    elif verdict == "LEGITIMATE":
        confidence = max(legit_probability, 1.0 - scam_probability, 0.85)
    elif verdict == "SUSPICIOUS":
        confidence = max(suspicious_probability, 0.65)
    else:
        confidence = misinformation_probability

    return {
        "message": clean_message,
        "verdict": verdict,
        "scam_probability": round(scam_probability, 4),
        "confidence": round(float(confidence), 4),
        "scam_type": scam_type,
        "risk_score": score,
        "risk_level": level,
        "indicators": indicators,
        "indicator_score": indicator_score,
        "reasons": reasons,
        "recommended_action": recommended_action(level, verdict),
    }