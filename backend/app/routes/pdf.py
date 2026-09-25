from fastapi import APIRouter,UploadFile,File

router=APIRouter(prefix="/api/v1/detect",tags=["PDF"])

@router.post("/pdf")
async def detect_pdf(file:UploadFile=File(...)):
    return {
        "status":"received",
        "filename":file.filename,
        "message":"PDF detection pipeline will be connected in Module 10"
    }