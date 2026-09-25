from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..mod7 import explain_message
from ..services.pdf_detector import MAX_PDF_SIZE_BYTES, PDFDetectionError, analyze_pdf
from ..services.url_detector import analyze_url

router = APIRouter(prefix="/api/v1/detect", tags=["Unified"])


@router.post("/unified")
async def detect_unified(
    message: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Unified scam detection endpoint accepting any combination of:
    - Text message (form field 'message')
    - Standalone URL (form field 'url')
    - Document attachment (file field 'file', PDF format)
    """
    if not message and not url and not file:
        raise HTTPException(
            status_code=400,
            detail="At least one input (message, url, or file) must be provided for analysis.",
        )

    results = []

    # 1. Inspect URL if provided
    if url and url.strip():
        try:
            url_res = analyze_url(url.strip())
            results.append(("url", url_res))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"URL error: {e}")

    # 2. Inspect PDF if provided
    if file:
        try:
            pdf_bytes = await file.read(MAX_PDF_SIZE_BYTES + 1)
            pdf_res = analyze_pdf(
                pdf_bytes,
                filename=file.filename,
                content_type=file.content_type,
            )
            results.append(("pdf", pdf_res))
        except PDFDetectionError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # 3. Inspect Text Message if provided
    if message and message.strip():
        msg_res = explain_message(message.strip(), metadata={"source": "text"})
        results.append(("text", msg_res))

    # Single-modality shortcut
    if len(results) == 1:
        source_type, single_verdict = results[0]
        single_verdict["source_type"] = source_type.upper()
        return single_verdict

    # Multi-modality fusion: pick the most severe security posture
    priority_order = {"SCAM": 3, "SUSPICIOUS": 2, "FALSE_MISINFORMATION": 1, "LEGITIMATE": 0}
    highest_priority = -1
    master_result = None

    for source_name, item in results:
        v = item.get("verdict_result", {}).get("verdict", item.get("verdict", "LEGITIMATE"))
        p = priority_order.get(v, 0)
        if p > highest_priority:
            highest_priority = p
            master_result = item

    master_result["source_type"] = "UNIFIED_MULTIMODAL"
    return master_result