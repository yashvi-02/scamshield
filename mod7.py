"""
MODULE 7 - Explainable Verdict Engine

Converts the deterministic output of mod6.py into a stable human-readable
security verdict. This module does not call an LLM and does not change the
model verdict, score, risk level, or scam type.
"""

from mod6 import analyze_message


INDICATOR_EXPLANATIONS = {
    "suspicious_url": "Contains an external or suspicious link",
    "phone_number": "Contains a phone number that may need verification",
    "otp_request": "Requests an OTP or verification code",
    "upi_or_wallet": "Mentions UPI or a digital wallet",
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


def _validate_result(risk_result):
    required = {
        "verdict",
        "scam_probability",
        "scam_type",
        "risk_score",
        "risk_level",
        "indicators",
    }
    missing = required.difference(risk_result)
    if missing:
        raise ValueError(f"Risk result is missing fields: {sorted(missing)}")


def build_verdict(risk_result):
    """Build deterministic structured explanation from a Module 6 result."""
    _validate_result(risk_result)
    verdict = str(risk_result["verdict"]).upper()
    level = str(risk_result["risk_level"]).upper()
    indicators = risk_result["indicators"]
    is_scam = verdict == "SCAM"

    explanations = [
        INDICATOR_EXPLANATIONS[name]
        for name, detected in indicators.items()
        if detected and name in INDICATOR_EXPLANATIONS
    ]
    actions = [
        INDICATOR_ACTIONS[name]
        for name, detected in indicators.items()
        if detected and name in INDICATOR_ACTIONS
    ]

    if is_scam:
        actions.insert(0, "Do not reply or continue the conversation")
        actions.append("Verify through the official organization application or website")
    elif not explanations:
        explanations.append("No strong scam indicators were detected")
        actions.append("Remain cautious with unexpected requests")

    return {
        "verdict": verdict,
        "headline": f"{verdict} - {level} RISK",
        "confidence_percent": round(float(risk_result["scam_probability"]) * 100, 2),
        "type": TYPE_DISPLAY_NAMES.get(
            risk_result["scam_type"], risk_result["scam_type"]
        ),
        "risk_score": risk_result["risk_score"],
        "why": explanations,
        "recommended_actions": list(dict.fromkeys(actions)),
        "indicators": dict(indicators),
    }


def format_verdict(verdict_result):
    """Render the structured verdict as deterministic plain text."""
    type_name = verdict_result["type"] or "Not applicable"
    lines = [
        verdict_result["headline"],
        f"Confidence: {verdict_result['confidence_percent']:.2f}%",
        f"Type: {type_name}",
        "",
        "Why:",
    ]
    lines.extend(f"- {reason}" for reason in verdict_result["why"])
    lines.extend(["", "Recommended action:"])
    lines.extend(f"- {action}" for action in verdict_result["recommended_actions"])
    return "\n".join(lines)


def explain_message(message, metadata=None):
    """Run Module 6 and return both structured and formatted explanations."""
    risk_result = analyze_message(message, metadata)
    verdict_result = build_verdict(risk_result)
    return {
        "risk_result": risk_result,
        "verdict_result": verdict_result,
        "text": format_verdict(verdict_result),
    }


if __name__ == "__main__":
    example = (
        "Your bank account will be blocked today. Share your OTP and UPI PIN "
        "immediately at https://verify-account.example.com to avoid arrest."
    )
    print(explain_message(example)["text"])