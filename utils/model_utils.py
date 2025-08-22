import numpy as np
import pickle
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from typing import Dict, List, Tuple, Optional
import os

from config import settings

logger = logging.getLogger(__name__)

class ModelUtils:
    @staticmethod
    def save_model(model, scaler, model_path: str, scaler_path: str):
        """Save trained model and scaler to disk"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save scaler
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Scaler saved to {scaler_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise e
    
    @staticmethod
    def load_model(model_path: str, scaler_path: str) -> Tuple[Optional[object], Optional[object]]:
        """Load model and scaler from disk"""
        try:
            model = None
            scaler = None
            
            # Load model
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info(f"Model loaded from {model_path}")
            
            # Load scaler
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                logger.info(f"Scaler loaded from {scaler_path}")
            
            return model, scaler
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None, None
    
    @staticmethod
    def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Classification report
            class_report = classification_report(
                y_test, y_pred, 
                target_names=settings.EMOTION_LABELS,
                output_dict=True
            )
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Per-class accuracy
            per_class_accuracy = {}
            for i, emotion in enumerate(settings.EMOTION_LABELS):
                if i < len(conf_matrix):
                    true_positive = conf_matrix[i, i]
                    total_actual = np.sum(conf_matrix[i, :])
                    if total_actual > 0:
                        per_class_accuracy[emotion] = true_positive / total_actual
                    else:
                        per_class_accuracy[emotion] = 0.0
            
            # Average confidence
            avg_confidence = np.mean(np.max(y_pred_proba, axis=1))
            
            results = {
                "accuracy": accuracy,
                "classification_report": class_report,
                "confusion_matrix": conf_matrix.tolist(),
                "per_class_accuracy": per_class_accuracy,
                "average_confidence": avg_confidence,
                "total_samples": len(y_test)
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def cross_validate_model(model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """Perform cross-validation"""
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            
            # Stratified K-Fold cross-validation
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            
            # Calculate cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
            
            results = {
                "cv_scores": cv_scores.tolist(),
                "mean_accuracy": np.mean(cv_scores),
                "std_accuracy": np.std(cv_scores),
                "cv_folds": cv
            }
            
            logger.info(f"Cross-validation completed: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def plot_confusion_matrix(conf_matrix: np.ndarray, save_path: str = None) -> str:
        """Generate confusion matrix plot"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                conf_matrix,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=settings.EMOTION_LABELS,
                yticklabels=settings.EMOTION_LABELS
            )
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Confusion matrix saved to {save_path}")
                return save_path
            else:
                # Return base64 encoded image if no save path provided
                import base64
                import io
                
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                return f"data:image/png;base64,{image_base64}"
                
        except Exception as e:
            logger.error(f"Error creating confusion matrix plot: {str(e)}")
            return ""
    
    @staticmethod
    def get_feature_importance(model, feature_names: List[str] = None) -> Dict:
        """Get feature importance from trained model"""
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # Create feature names if not provided
                if feature_names is None:
                    feature_names = [f"feature_{i}" for i in range(len(importances))]
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                feature_importance = {
                    "features": [feature_names[i] for i in indices],
                    "importances": importances[indices].tolist(),
                    "top_10_features": {
                        feature_names[indices[i]]: float(importances[indices[i]])
                        for i in range(min(10, len(indices)))
                    }
                }
                
                return feature_importance
            else:
                return {"error": "Model does not support feature importance"}
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {"error": str(e)}
    
    @staticmethod
    def create_model_summary(model, scaler, evaluation_results: Dict) -> Dict:
        """Create comprehensive model summary"""
        try:
            summary = {
                "model_type": type(model).__name__,
                "model_parameters": model.get_params() if hasattr(model, 'get_params') else {},
                "scaler_type": type(scaler).__name__ if scaler else None,
                "performance_metrics": {
                    "accuracy": evaluation_results.get("accuracy", 0.0),
                    "average_confidence": evaluation_results.get("average_confidence", 0.0),
                    "total_samples": evaluation_results.get("total_samples", 0)
                },
                "per_class_performance": evaluation_results.get("per_class_accuracy", {}),
                "feature_dimension": settings.N_MFCC * 3 + settings.N_CHROMA + 4 + 20 + 5,  # Total features
                "emotion_categories": settings.EMOTION_LABELS,
                "training_config": {
                    "sample_rate": settings.SAMPLE_RATE,
                    "duration": settings.DURATION,
                    "n_mfcc": settings.N_MFCC,
                    "n_chroma": settings.N_CHROMA,
                    "n_mel": settings.N_MEL
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating model summary: {str(e)}")
            return {"error": str(e)}
