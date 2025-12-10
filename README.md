# Overview

The Speech Emotion Recognition System is a FastAPI-based web application that analyzes audio input to detect and classify human emotions in real-time. The system supports both file uploads and live microphone recording, providing immediate emotion classification with confidence scores. It uses machine learning techniques with audio feature extraction (MFCC, chroma, spectral features) and a Random Forest classifier to identify seven emotion categories: neutral, happy, sad, angry, fear, disgust, and surprise.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Backend Architecture
- **Framework**: FastAPI for REST API endpoints and WebSocket support
- **Audio Processing**: librosa and pydub for audio file handling and feature extraction
- **Machine Learning**: scikit-learn with Random Forest classifier for emotion prediction
- **Feature Extraction**: Comprehensive audio feature extraction including MFCC, chroma, mel spectrograms, and spectral characteristics
- **Real-time Processing**: WebSocket connections for live audio streaming and emotion detection

## Frontend Architecture
- **Technology**: Vanilla JavaScript with HTML5 and CSS3
- **Audio Handling**: Web Audio API for microphone access and real-time audio processing
- **Visualization**: Chart.js for displaying emotion probabilities and real-time emotion tracking
- **UI Components**: Responsive design with cards, status bars, and interactive controls

## Data Processing Pipeline
- **Audio Input**: Supports multiple formats (.wav, .mp3, .m4a, .flac) with file size limits
- **Feature Extraction**: Multi-dimensional feature vectors combining acoustic and spectral features
- **Model Training**: Synthetic data initialization with capability for real dataset training
- **Prediction Pipeline**: Standardized feature scaling followed by classification with probability outputs

## Configuration Management
- **Centralized Settings**: Single config.py file managing audio processing parameters, model paths, and API settings
- **Environment Variables**: Support for DEBUG mode and logging levels
- **Feature Parameters**: Configurable MFCC coefficients, FFT window sizes, and sample rates

## File Structure Organization
- **Modular Design**: Separated concerns with dedicated modules for audio processing, feature extraction, and model management
- **Static Assets**: Self-contained frontend with CSS, JavaScript, and HTML files
- **Model Persistence**: Pickle-based model and scaler storage for trained classifier reuse

# External Dependencies

## Core Libraries
- **FastAPI**: Web framework for API development and WebSocket support
- **librosa**: Professional audio analysis and feature extraction library
- **scikit-learn**: Machine learning algorithms and preprocessing tools
- **numpy**: Numerical computing for audio data manipulation
- **pydub**: Audio file format conversion and processing

## Frontend Dependencies
- **Chart.js**: Data visualization for emotion probability charts
- **Feather Icons**: Icon library for UI elements
- **Web Audio API**: Browser-native audio recording and processing

## Development Tools
- **uvicorn**: ASGI server for FastAPI application deployment
- **logging**: Built-in Python logging for system monitoring and debugging

## Audio Processing Stack
- **Sample Rate**: 22.05 kHz standardized processing
- **Feature Extraction**: MFCC (13 coefficients), chroma (12 features), mel spectrograms (128 bands)
- **File Support**: Multiple audio formats with automatic conversion capabilities
- 
