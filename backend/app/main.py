from fastapi import FastAPI,HTTPException
from .database import test_connection
from .routes.message import router as message_router
from .routes.pdf import router as pdf_router

app.include_router(pdf_router)
app=FastAPI(title="ScamShield API")

app.include_router(message_router)

@app.get("/")
def home():
    return {"message":"ScamShield Backend is running"}

@app.get("/api/v1/health")
def health():
    return {"status":"OK","service":"ScamShield API"}

@app.get("/db-health")
def db_health():
    try:
        database,user,port=test_connection()
        return {
            "status":"connected",
            "database":database,
            "user":user,
            "port":port
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Database connection failed: {e}"
        )