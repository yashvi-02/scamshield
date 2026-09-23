from fastapi import APIRouter

router=APIRouter(prefix="/api/v1/detect",tags=["Detection"])

@router.post("/message")
def detect_message(data:dict):
    return {
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