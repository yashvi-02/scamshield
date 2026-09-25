from pathlib import Path
from mod6 import analyze_message

INDICATOR_EXPLANATIONS = {
    "suspicious_url": "Contains an external or suspicious link",
    "phone_number": "Contains a phone number that may need verification",
    "otp_request": "Requests an OTP or verification code",
    "upi_or_wallet": "Requests UPI or digital wallet information",
    "money_request": "Requests money, payment, transfer, or recharge",
    "authority_impersonation": "Uses government, bank, police, or authority language",
    "urgent_language": "Creates urgency or pressure to act quickly",
    "credential_request": "Requests sensitive credentials or account information",
    "threat_or_consequence": "Uses a threat or consequence to pressure the recipient",
}

INDICATOR_ACTIONS = {
    "suspicious_url": "Do not click the link",
    "phone_number": "Do not call an unverified number",
    "otp_request": "Do not share your OTP or verification code",
    "upi_or_wallet": "Do not share your UPI PIN or wallet credentials",
    "money_request": "Do not transfer money or make a payment",
    "authority_impersonation": "Verify through the official organization website or application",
    "urgent_language": "Do not make a rushed decision",
    "credential_request": "Do not share passwords, PINs, CVV, or account details",
    "threat_or_consequence": "Do not comply with threats; contact the official authority directly",
}

TYPE_DISPLAY_NAMES = {
    "KYC_FRAUD": "KYC Fraud",
    "OTP_THEFT": "OTP Theft",
    "UPI_FRAUD": "UPI Fraud",
    "BANK_IMPERSONATION": "Bank Impersonation",
    "POLICE_IMPERSONATION": "Police Impersonation",
    "DIGITAL_ARREST": "Digital Arrest",
    "JOB_SCAM": "Job Scam",
    "INVESTMENT_SCAM": "Investment Scam",
    "COURIER_FRAUD": "Courier Fraud",
    "LOAN_SCAM": "Loan Scam",
    "LOTTERY_SCAM": "Lottery Scam",
    "PHISHING": "Phishing",
    "ACCOUNT_TAKEOVER": "Account Takeover",
    "ELECTRICITY_SCAM": "Electricity Scam",
    "TAX_REFUND_SCAM": "Tax Refund Scam",
    "SCHOLARSHIP_SCAM": "Scholarship Scam",
    "CHARITY_SCAM": "Charity Scam",
}

def build_verdict(risk_result: dict) -> dict:
    verdict = risk_result["verdict"]
    level = risk_result["risk_level"]
    indicators = risk_result.get("indicators", {})

    detected_indicators = [name for name, det in indicators.items() if det]

    if detected_indicators:
        explanations = [
            INDICATOR_EXPLANATIONS[name]
            for name in detected_indicators
            if name in INDICATOR_EXPLANATIONS
        ]
        actions = [
            INDICATOR_ACTIONS[name]
            for name in detected_indicators
            if name in INDICATOR_ACTIONS
        ]
    else:
        explanations = []
        actions = []

    if verdict == "SCAM":
        actions.insert(0, "Do not reply or continue the conversation")
        actions.append("Verify through the official organization application or website")
    elif verdict == "LEGITIMATE":
        explanations = ["No strong scam indicators were detected"]
        actions = ["No action is required. Standard caution is sufficient."]
    elif verdict == "SUSPICIOUS":
        if not explanations:
            explanations = ["The message contains signals that require verification"]
        actions.insert(0, "Do not share sensitive information until verified")
        actions.append("Verify the sender through an official channel")
    elif verdict == "FALSE_MISINFORMATION":
        if not explanations:
            explanations = ["The message may contain misleading or unverified information"]
        actions.insert(0, "Do not forward the message without verification")
        actions.append("Verify the information through an official source")

    scam_type_key = risk_result.get("scam_type")
    type_display = TYPE_DISPLAY_NAMES.get(scam_type_key, scam_type_key) if scam_type_key else "Not applicable"

    return {
        "verdict": verdict,
        "headline": f"{verdict} - {level} RISK",
        "confidence_percent": round(float(risk_result.get("confidence", 0.0)) * 100, 2),
        "type": type_display,
        "risk_score": risk_result.get("risk_score", 0.0),
        "why": list(dict.fromkeys(explanations)),
        "recommended_actions": list(dict.fromkeys(actions)),
        "indicators": dict(indicators),
    }

def format_verdict(result: dict) -> str:
    lines = [
        "[SCAMSHIELD ANALYSIS]",
        "",
        f"Verdict: {result['verdict']}",
        f"Risk Level: {result['headline'].split(' - ')[-1]}",
        f"Confidence: {result['confidence_percent']:.2f}%",
    ]

    if result["verdict"] == "SCAM":
        lines.append(f"Type: {result['type']}")

    lines.extend(["", "Why:"])
    lines.extend(f"- {x}" for x in result["why"])

    lines.extend(["", "Recommended Action:"])
    lines.extend(f"- {x}" for x in result["recommended_actions"])

    return "\n".join(lines)

def explain_message(message: str, metadata: dict = None) -> dict:
    risk_result = analyze_message(message, metadata)
    verdict_result = build_verdict(risk_result)
    return {
        "risk_result": risk_result,
        "verdict_result": verdict_result,
        "text": format_verdict(verdict_result),
    }

if __name__ == "__main__":
    test_cases = [
        "hey, party today at 8",
        "Your bank account will be blocked today. Verify your KYC immediately by sharing your OTP and UPI PIN.",
    ]
    for tc in test_cases:
        print("=" * 70)
        res = explain_message(tc)
        print(res["text"])
    print("=" * 70)