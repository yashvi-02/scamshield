from fastapi import APIRouter, File, HTTPException, UploadFile

from ..services.pdf_detector import (
    MAX_PDF_SIZE_BYTES,
    PDFDetectionError,
    analyze_pdf,
)

router = APIRouter(prefix="/api/v1/detect", tags=["PDF"])


@router.post("/pdf")
async def detect_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read(MAX_PDF_SIZE_BYTES + 1)
        return analyze_pdf(
            pdf_bytes,
            filename=file.filename,
            content_type=file.content_type,
        )
    except PDFDetectionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc