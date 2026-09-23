def detect_message(message:str):
    result={
        "verdict":"SCAM",
        "risk_level":"HIGH",
        "confidence":0.94,
        "scam_type":"KYC_FRAUD",
        "evidence":[
            "OTP request",
            "Urgency",
            "Authority impersonation"
        ],
        "recommended_action":"Do not share OTP"
    }

    return result