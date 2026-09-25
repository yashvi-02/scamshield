from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.url_detector import URLDetectionError, analyze_url


router = APIRouter(prefix="/api/v1/detect", tags=["URL"])


class URLRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)


@router.post("/url")
def detect_url(data: URLRequest):
    try:
        return analyze_url(data.url)
    except URLDetectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
