from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.detector import detect_message as run_detector

router = APIRouter(
    prefix="/api/v1/detect",
    tags=["Detection"]
)


class MessageRequest(BaseModel):
    message: str


@router.post("/message")
def detect_message(data: MessageRequest):
    if not data.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

    return run_detector(data.message)