from fastapi import APIRouter, Form, Response
from typing import Optional
from mod7 import explain_message

router = APIRouter(tags=["WhatsApp"])

@router.post("/whatsapp")
async def receive_whatsapp(
    Body: Optional[str] = Form(None),
    From: Optional[str] = Form(None),
    MediaUrl0: Optional[str] = Form(None),
    NumMedia: Optional[str] = Form("0")
):
    incoming_text = (Body or "").strip()

    if not incoming_text and int(NumMedia or 0) > 0:
        reply_body = (
            "[SCAMSHIELD ANALYSIS]\n\n"
            "Verdict: SUSPICIOUS\n"
            "Risk Level: MEDIUM RISK\n"
            "Confidence: 75.00%\n\n"
            "Why:\n"
            "- Received media/document attachment\n\n"
            "Recommended Action:\n"
            "- Media inspection module active. Forward text for immediate classification."
        )
    elif not incoming_text:
        reply_body = (
            "[SCAMSHIELD ANALYSIS]\n\n"
            "Verdict: UNKNOWN\n"
            "Risk Level: LOW RISK\n"
            "Confidence: 0.00%\n\n"
            "Why:\n"
            "- Empty message received\n\n"
            "Recommended Action:\n"
            "- Please send or forward a text message to verify."
        )
    else:
        analysis = explain_message(incoming_text)
        reply_body = analysis["text"]

    twiml_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{reply_body}</Message>
</Response>"""

    return Response(content=twiml_xml, media_type="application/xml")