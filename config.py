import os
from typing import List

class Settings:
    # Audio processing settings
    SAMPLE_RATE: int = 22050
    DURATION: float = 3.0  # seconds
    MIN_AUDIO_LENGTH: int = int(SAMPLE_RATE * 0.5)  # 0.5 seconds minimum
    
    # Emotion categories
    EMOTION_LABELS: List[str] = [
        "neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"
    ]
    
    # Model settings
    MODEL_PATH: str = "models/trained_model.pkl"
    SCALER_PATH: str = "models/scaler.pkl"
    
    # Feature extraction settings
    N_MFCC: int = 13
    N_CHROMA: int = 12
    N_MEL: int = 128
    HOP_LENGTH: int = 512
    N_FFT: int = 2048
    
    # File upload settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".wav", ".mp3", ".m4a", ".flac", ".webm", ".ogg"]
    
    # API settings
    API_TIMEOUT: int = 30
    
    # Environment variables
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
