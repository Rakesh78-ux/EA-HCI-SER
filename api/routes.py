from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import tempfile
import os
import logging
from typing import Optional
import numpy as np

from models.emotion_classifier import EmotionClassifier
from services.audio_processor import AudioProcessor
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances will be imported from main
emotion_classifier = None
audio_processor = None

@router.post("/predict/file")
async def predict_from_file(
    file: UploadFile = File(...),
    return_probabilities: bool = Form(True)
):
    """Predict emotion from uploaded audio file"""
    try:
        # Import global instances
        from main import emotion_classifier, audio_processor
        # Validate file size
        if file.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Process audio file
        result = await process_audio_file(file_content, file_ext, return_probabilities)

        # Log full prediction for debugging
        try:
            logger.info("PREDICT_FROM_FILE result: predicted=%s confidence=%.4f emotions=%s",
                    result.get("predicted_emotion"),
                    float(result.get("confidence", 0.0)),
                    result.get("emotion_probabilities") or result.get("emotions") or {})
            if "message" in result:
                logger.info("PREDICT_FROM_FILE message: %s", result["message"])
        except Exception:
            logger.exception("Failed to log prediction result")

        return JSONResponse(content=result)


        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_from_file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/predict/audio")
async def predict_from_audio_data(
    audio_data: bytes,
    sample_rate: int = 22050,
    return_probabilities: bool = True
):
    """Predict emotion from raw audio data"""
    try:
        # Import global instances
        from main import emotion_classifier, audio_processor
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        
        if len(audio_array) == 0:
            raise HTTPException(status_code=400, detail="Empty audio data")
        
        # Process audio
        processed_audio = audio_processor.preprocess_audio(audio_array, sample_rate)
        
        prediction = emotion_classifier.predict(processed_audio, sample_rate)

        # Log full prediction for debugging
        try:
            logger.info("PREDICT_FROM_AUDIO prediction: predicted=%s confidence=%.4f emotions=%s",
                        prediction.get("predicted_emotion"),
                        float(prediction.get("confidence", 0.0)),
                        prediction.get("emotions", {}))
        except Exception:
            logger.exception("Failed to log prediction")

        # Format response
        response = {
            "predicted_emotion": prediction["predicted_emotion"],
            "confidence": prediction["confidence"],
            "status": prediction.get("status", "success")
        }

        if return_probabilities:
            response["emotion_probabilities"] = prediction.get("emotions", {})

        
        if "message" in prediction:
            response["message"] = prediction["message"]
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_from_audio_data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/emotions")
async def get_supported_emotions():
    """Get list of supported emotion categories"""
    return {
        "emotions": settings.EMOTION_LABELS,
        "total_emotions": len(settings.EMOTION_LABELS)
    }

@router.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    try:
        # Import global instances
        from main import emotion_classifier
        model_info = {
            "model_type": "RandomForestClassifier",
            "is_trained": emotion_classifier.is_trained,
            "supported_emotions": settings.EMOTION_LABELS,
            "feature_dimension": emotion_classifier.feature_extractor.get_feature_dimension(),
            "sample_rate": settings.SAMPLE_RATE,
            "audio_duration": settings.DURATION
        }
        
        return JSONResponse(content=model_info)
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Import global instances
        from main import emotion_classifier, audio_processor
        # Check if model is loaded
        if not emotion_classifier.is_trained:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "Model not loaded"}
            )
        
        # Check if audio processor is initialized
        if not audio_processor.initialized:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "Audio processor not initialized"}
            )
        
        return {
            "status": "healthy",
            "model_loaded": True,
            "audio_processor_ready": True,
            "supported_emotions": len(settings.EMOTION_LABELS)
        }
        
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )

async def process_audio_file(file_content: bytes, file_ext: str, return_probabilities: bool = True) -> dict:
    """Process uploaded audio file and return emotion prediction"""
    try:
        # Import global instances
        from main import emotion_classifier, audio_processor
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Load audio file
            audio_result = audio_processor.load_audio_file(temp_file_path)
            
            if audio_result is None:
                return {
                    "error": "Failed to load audio file",
                    "predicted_emotion": "neutral",
                    "confidence": 0.0,
                    "status": "error"
                }
            
            audio_data, sample_rate = audio_result

            # DEBUG LOG — raw audio details from file route
            try:
                logger.info(
                    "ROUTE process_audio_file: received audio samples=%d sample_rate=%s dtype=%s",
                    len(audio_data),
                    sample_rate,
                    getattr(audio_data, "dtype", "unknown")
                )
                logger.info(
                    "ROUTE process_audio_file: audio min/max/mean = %.6f / %.6f / %.6f",
                    float(audio_data.min()),
                    float(audio_data.max()),
                    float(audio_data.mean())
                )
            except Exception:
                logger.exception("Failed to log raw audio stats")

            
            # Preprocess audio
            processed_audio = audio_processor.preprocess_audio(audio_data, sample_rate)
            
            # Get prediction
            prediction = emotion_classifier.predict(processed_audio, sample_rate)
            
            # Format response
            response = {
                "predicted_emotion": prediction["predicted_emotion"],
                "confidence": prediction["confidence"],
                "status": prediction.get("status", "success"),
                "audio_info": {
                    "duration": len(processed_audio) / sample_rate,
                    "sample_rate": sample_rate,
                    "samples": len(processed_audio)
                }
            }
            
            if return_probabilities:
                response["emotion_probabilities"] = prediction["emotions"]
            
            if "message" in prediction:
                response["message"] = prediction["message"]
            
            return response
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        logger.error(f"Error processing audio file: {str(e)}")
        return {
            "error": f"Processing failed: {str(e)}",
            "predicted_emotion": "neutral",
            "confidence": 0.0,
            "status": "error"
        }
