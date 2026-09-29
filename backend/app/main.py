from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import websocket

app = FastAPI(
    title="AI-Powered Real-Time ISL Translator",
    description="Backend API for recognizing Indian Sign Language gestures in real-time.",
    version="1.0.0"
)

# CORS configuration
origins = settings.cors_origins.split(",") if settings.cors_origins else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket.router, prefix="/ws", tags=["websocket"])

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "isl-translator-backend"
    }
