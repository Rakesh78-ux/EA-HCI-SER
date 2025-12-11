import pickle
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from typing import Dict
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
            if os.path.exists(settings.MODEL_PATH) and os.path.exists(settings.SCALER_PATH):
                with open(settings.MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(settings.SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("Pre-trained model loaded successfully")
            else:
                logger.info("No pre-trained model found. Initializing new model...")
                self._initialize_default_model()

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self._initialize_default_model()

    def _initialize_default_model(self):
        """Initialize a basic model with synthetic training data for demonstration"""
        logger.info("Initializing default model with synthetic data...")

        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )
        self.scaler = StandardScaler()

        n_samples = 1000
        n_features = self.feature_extractor.get_feature_dimension()

        X_synthetic = np.random.randn(n_samples, n_features)
        y_synthetic = np.random.randint(0, len(settings.EMOTION_LABELS), n_samples)

        for i in range(len(settings.EMOTION_LABELS)):
            start = i * (n_samples // len(settings.EMOTION_LABELS))
            end = (i + 1) * (n_samples // len(settings.EMOTION_LABELS))
            if end > n_samples:
                end = n_samples

            X_synthetic[start:end] += np.random.randn(1, n_features) * 0.5
            y_synthetic[start:end] = i

        X_scaled = self.scaler.fit_transform(X_synthetic)
        self.model.fit(X_scaled, y_synthetic)

        self.is_trained = True
        self._save_model()

        logger.info("Default model initialized and trained")

    def predict(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Predict emotion from audio data (with debug logging and extra diagnostics)."""
        try:
            if not self.is_trained:
                raise Exception("Model not trained or loaded")

            # --- Extract features ---
            features = self.feature_extractor.extract_features(audio_data, sample_rate)

            # If extractor failed or returned nothing, give clear reason.
            if features is None or len(features) == 0:
                return self._get_default_prediction("Feature extraction returned empty/None")

            # Ensure features are numeric array
            features = np.asarray(features, dtype=np.float32).reshape(1, -1)

            # Quick diagnostics on features
            try:
                fmin = float(np.min(features))
                fmax = float(np.max(features))
                fmean = float(np.mean(features))
                fstd = float(np.std(features))
                logger.info("FEATURES shape=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
                            features.shape, fmin, fmax, fmean, fstd)
            except Exception:
                logger.info("FEATURES diagnostics unavailable")

            # If features are all zeros or zero variance, fail early with reason
            if np.allclose(features, 0.0) or np.isclose(np.std(features), 0.0):
                return self._get_default_prediction("Extracted features are all zeros or have zero variance")

            # Scale features (if scaler fails, log and continue with raw features)
            try:
                features_scaled = self.scaler.transform(features)
            except Exception as e:
                logger.exception("Scaler transform failed: %s", e)
                features_scaled = features  # fallback

            # Predict probabilities (protect against errors)
            try:
                probabilities = self.model.predict_proba(features_scaled)[0]
            except Exception as e:
                logger.exception("predict_proba failed: %s", e)
                return self._get_default_prediction("predict_proba failed")

            # Basic sanity check on probabilities
            if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                return self._get_default_prediction("Model returned invalid probabilities")

            # Normalize probs (just in case)
            probs_sum = float(np.sum(probabilities))
            if probs_sum <= 0 or np.isclose(probs_sum, 0.0):
                return self._get_default_prediction("Model probabilities sum to zero")
            probabilities = probabilities / probs_sum

            # Determine predicted index and confidence
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])

            # Map labels robustly
            model_labels = getattr(self.model, "classes_", None)
            if model_labels is not None and len(model_labels) == len(probabilities):
                # model_labels might be ints (class indices) or strings
                try:
                    if all(isinstance(x, (int, np.integer)) for x in model_labels):
                        predicted_emotion = settings.EMOTION_LABELS[int(model_labels[predicted_index])]
                    else:
                        predicted_emotion = str(model_labels[predicted_index])
                except Exception:
                    predicted_emotion = settings.EMOTION_LABELS[predicted_index]
            else:
                predicted_emotion = settings.EMOTION_LABELS[predicted_index]

            # Build probability dictionary mapping names -> prob
            emotion_probs = {}
            if model_labels is not None and len(model_labels) == len(probabilities):
                for i, lbl in enumerate(model_labels):
                    if isinstance(lbl, (int, np.integer)):
                        emotion_name = settings.EMOTION_LABELS[int(lbl)]
                    else:
                        emotion_name = str(lbl)
                    emotion_probs[emotion_name] = float(probabilities[i])
            else:
                for i, emotion in enumerate(settings.EMOTION_LABELS):
                    # safe guard: if index out of range, use placeholder name
                    name = emotion if i < len(settings.EMOTION_LABELS) else f"label_{i}"
                    emotion_probs[name] = float(probabilities[i])

            logger.info("PREDICTION probs: %s | predicted=%s | conf=%.4f",
                        emotion_probs, predicted_emotion, confidence)

            # If model is uncertain (low confidence), do not silently return neutral — mark as low_confidence
            LOW_CONFIDENCE_THRESHOLD = getattr(settings, "LOW_CONFIDENCE_THRESHOLD", 0.25)
            low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

            result = {
                "emotions": emotion_probs,
                "predicted_emotion": predicted_emotion,
                "confidence": confidence,
                "status": "success"
            }
            if low_confidence:
                result["low_confidence"] = True
                result["note"] = f"low confidence (threshold {LOW_CONFIDENCE_THRESHOLD})"

            return result

        except Exception as e:
            logger.exception("Prediction error")
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
        """Train the model"""
        try:
            logger.info("Starting model training...")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model.fit(X_train_scaled, y_train)

            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)

            self.is_trained = True
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
            os.makedirs(os.path.dirname(settings.MODEL_PATH), exist_ok=True)

            with open(settings.MODEL_PATH, 'wb') as f:
                pickle.dump(self.model, f)

            with open(settings.SCALER_PATH, 'wb') as f:
                pickle.dump(self.scaler, f)

            logger.info("Model and scaler saved successfully")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise e
