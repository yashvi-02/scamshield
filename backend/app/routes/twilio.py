from fastapi import APIRouter, Request
from fastapi.responses import Response

from ..services.detector import detect_message

router = APIRouter(
    prefix="/api/v1/twilio",
    tags=["Twilio"]
)


@router.post("/whatsapp")
async def whatsapp_webhook(request: Request):
    form = await request.form()

    message = form.get("Body", "").strip()

    if not message:
        return Response(
            content="""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>Please send a message for ScamShield analysis.</Message>
</Response>""",
            media_type="application/xml"
        )

    try:
        result = detect_message(message)

        # The existing Module 7 response already contains
        # a WhatsApp-friendly formatted text.
        response_text = result.get("text", "Unable to analyze the message.")

    except Exception as e:
        response_text = (
            "ScamShield could not analyze this message right now.\n"
            "Please try again."
        )

        print(f"Twilio webhook error: {e}")

    # Escape XML-sensitive characters
    response_text = (
        response_text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Message>{response_text}</Message>
</Response>"""

    return Response(
        content=twiml,
        media_type="application/xml"
    )