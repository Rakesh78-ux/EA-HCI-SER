from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import logging
import json
import asyncio
from typing import List
import numpy as np

from models.emotion_classifier import EmotionClassifier
from services.audio_processor import AudioProcessor
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances (must be created before importing routes)
emotion_classifier = EmotionClassifier()
audio_processor = AudioProcessor()

# Import routes after creating global instances
from api.routes import router

app = FastAPI(
    title="Speech Emotion Recognition System",
    description="Real-time emotion recognition from audio input",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(router, prefix="/api")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return FileResponse("static/index.html")

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_bytes()
            
            try:
                # Process audio data
                audio_array = np.frombuffer(data, dtype=np.float32)
                
                if len(audio_array) > 0:
                    # Process and predict emotion
                    prediction = await process_audio_stream(audio_array)
                    
                    # Send prediction back to client
                    await manager.send_personal_message(
                        json.dumps(prediction), websocket
                    )
                    
            except Exception as e:
                logger.error(f"Error processing audio stream: {str(e)}")
                error_response = {
                    "error": "Audio processing failed",
                    "message": str(e)
                }
                await manager.send_personal_message(
                    json.dumps(error_response), websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket connection closed")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

async def process_audio_stream(audio_data: np.ndarray) -> dict:
    """Process streaming audio data and return emotion prediction"""
    try:
        # Ensure minimum audio length for processing
        if len(audio_data) < settings.MIN_AUDIO_LENGTH:
            return {
                "emotions": {},
                "predicted_emotion": "neutral",
                "confidence": 0.0,
                "message": "Audio too short for reliable prediction"
            }
        
        # Process audio and extract features
        processed_audio = audio_processor.preprocess_audio(audio_data, settings.SAMPLE_RATE)
        
        # Get emotion prediction
        prediction = emotion_classifier.predict(processed_audio, settings.SAMPLE_RATE)
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error in process_audio_stream: {str(e)}")
        return {
            "error": "Processing failed",
            "message": str(e),
            "emotions": {},
            "predicted_emotion": "neutral",
            "confidence": 0.0
        }

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    logger.info("Starting Speech Emotion Recognition System...")
    
    try:
        # Initialize emotion classifier
        emotion_classifier.load_model()
        logger.info("Emotion classifier loaded successfully")
        
        # Initialize audio processor
        audio_processor.initialize()
        logger.info("Audio processor initialized successfully")
        
        logger.info("System startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Speech Emotion Recognition System...")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
        log_level="info"
    )
