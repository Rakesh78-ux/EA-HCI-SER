import pickle
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict, Tuple, Optional
import os

from services.feature_extractor import FeatureExtractor
from config import settings

logger = logging.getLogger(__name__)

class EmotionClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        
    def load_model(self):
        """Load pre-trained model and scaler"""
        try:
            # Try to load existing model
            if os.path.exists(settings.MODEL_PATH) and os.path.exists(settings.SCALER_PATH):
                with open(settings.MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(settings.SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("Pre-trained model loaded successfully")
            else:
                # Initialize and train a basic model
                logger.info("No pre-trained model found. Initializing new model...")
                self._initialize_default_model()
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._initialize_default_model()
    
    def _initialize_default_model(self):
        """Initialize a basic model with synthetic training data for demonstration"""
        logger.info("Initializing default model with synthetic data...")
        
        # Initialize model and scaler
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.scaler = StandardScaler()
        
        # Create synthetic training data for initial model
        n_samples = 1000
        n_features = self.feature_extractor.get_feature_dimension()
        
        # Generate synthetic features
        X_synthetic = np.random.randn(n_samples, n_features)
        
        # Create balanced synthetic labels
        y_synthetic = np.random.randint(0, len(settings.EMOTION_LABELS), n_samples)
        
        # Add some structure to make it more realistic
        for i in range(len(settings.EMOTION_LABELS)):
            start_idx = i * (n_samples // len(settings.EMOTION_LABELS))
            end_idx = (i + 1) * (n_samples // len(settings.EMOTION_LABELS))
            if end_idx > n_samples:
                end_idx = n_samples
                
            # Modify features to have some emotion-specific characteristics
            X_synthetic[start_idx:end_idx] += np.random.randn(1, n_features) * 0.5
            y_synthetic[start_idx:end_idx] = i
        
        # Train the model
        X_scaled = self.scaler.fit_transform(X_synthetic)
        self.model.fit(X_scaled, y_synthetic)
        
        self.is_trained = True
        
        # Save the initialized model
        self._save_model()
        
        logger.info("Default model initialized and trained")
    
    def predict(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Predict emotion from audio data"""
        try:
            if not self.is_trained:
                raise Exception("Model not trained or loaded")
            
            # Extract features
            features = self.feature_extractor.extract_features(audio_data, sample_rate)
            
            if features is None or len(features) == 0:
                return self._get_default_prediction("Feature extraction failed")
            
            # Reshape for prediction
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Get predicted class
            predicted_class = np.argmax(probabilities)
            predicted_emotion = settings.EMOTION_LABELS[predicted_class]
            confidence = float(probabilities[predicted_class])
            
            # Create emotion probability dictionary
            emotion_probs = {}
            for i, emotion in enumerate(settings.EMOTION_LABELS):
                emotion_probs[emotion] = float(probabilities[i])
            
            return {
                "emotions": emotion_probs,
                "predicted_emotion": predicted_emotion,
                "confidence": confidence,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return self._get_default_prediction(f"Prediction failed: {str(e)}")
    
    def _get_default_prediction(self, message: str) -> Dict:
        """Return default prediction when processing fails"""
        default_probs = {emotion: 1.0 / len(settings.EMOTION_LABELS) 
                        for emotion in settings.EMOTION_LABELS}
        
        return {
            "emotions": default_probs,
            "predicted_emotion": "neutral",
            "confidence": 1.0 / len(settings.EMOTION_LABELS),
            "status": "error",
            "message": message
        }
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train the model with provided data"""
        try:
            logger.info("Starting model training...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            
            # Save model
            self._save_model()
            
            logger.info(f"Model training completed. Accuracy: {accuracy:.3f}")
            
            return {
                "accuracy": accuracy,
                "classification_report": classification_report(
                    y_test, y_pred, target_names=settings.EMOTION_LABELS
                )
            }
            
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            raise e
    
    def _save_model(self):
        """Save trained model and scaler"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)
            
            # Save model
            with open(settings.MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            with open(settings.SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            logger.info("Model and scaler saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise e
