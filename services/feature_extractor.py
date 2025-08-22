import numpy as np
import librosa
import logging
from typing import Optional, Tuple
from config import settings

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        self.feature_dimension = None
        self._calculate_feature_dimension()
    
    def _calculate_feature_dimension(self):
        """Calculate the total number of features extracted"""
        # MFCC features + delta + delta2
        mfcc_features = settings.N_MFCC * 3  # MFCC + delta + delta2
        
        # Chroma features
        chroma_features = settings.N_CHROMA
        
        # Mel spectrogram features (statistical features)
        mel_features = 4  # mean, std, skew, kurtosis
        
        # Spectral features
        spectral_features = 20  # Various spectral characteristics
        
        # Rhythm and tempo features
        rhythm_features = 5
        
        self.feature_dimension = (
            mfcc_features + chroma_features + mel_features + 
            spectral_features + rhythm_features
        )
    
    def get_feature_dimension(self) -> int:
        """Get the total number of features"""
        return self.feature_dimension
    
    def extract_features(self, audio_data: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Extract comprehensive audio features for emotion recognition"""
        try:
            # Ensure audio is not empty
            if len(audio_data) == 0:
                logger.warning("Empty audio data provided")
                return None
            
            # Resample if necessary
            if sample_rate != settings.SAMPLE_RATE:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=settings.SAMPLE_RATE
                )
                sample_rate = settings.SAMPLE_RATE
            
            features = []
            
            # 1. MFCC features
            mfcc_features = self._extract_mfcc_features(audio_data, sample_rate)
            features.extend(mfcc_features)
            
            # 2. Chroma features
            chroma_features = self._extract_chroma_features(audio_data, sample_rate)
            features.extend(chroma_features)
            
            # 3. Mel spectrogram features
            mel_features = self._extract_mel_features(audio_data, sample_rate)
            features.extend(mel_features)
            
            # 4. Spectral features
            spectral_features = self._extract_spectral_features(audio_data, sample_rate)
            features.extend(spectral_features)
            
            # 5. Rhythm and tempo features
            rhythm_features = self._extract_rhythm_features(audio_data, sample_rate)
            features.extend(rhythm_features)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return None
    
    def _extract_mfcc_features(self, audio_data: np.ndarray, sample_rate: int) -> list:
        """Extract MFCC features with delta and delta-delta"""
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=sample_rate,
                n_mfcc=settings.N_MFCC,
                n_fft=settings.N_FFT,
                hop_length=settings.HOP_LENGTH
            )
            
            # Calculate delta and delta-delta features
            delta_mfccs = librosa.feature.delta(mfccs)
            delta2_mfccs = librosa.feature.delta(mfccs, order=2)
            
            # Combine all MFCC features
            all_mfccs = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
            
            # Statistical features (mean, std) for each coefficient
            mfcc_features = []
            for i in range(all_mfccs.shape[0]):
                mfcc_features.append(np.mean(all_mfccs[i]))
                # Add std dev as well for more information
                if len(mfcc_features) < settings.N_MFCC * 3:
                    mfcc_features.append(np.std(all_mfccs[i]))
            
            # Ensure we have exactly N_MFCC * 3 features
            while len(mfcc_features) < settings.N_MFCC * 3:
                mfcc_features.append(0.0)
            
            return mfcc_features[:settings.N_MFCC * 3]
            
        except Exception as e:
            logger.error(f"MFCC extraction error: {str(e)}")
            return [0.0] * (settings.N_MFCC * 3)
    
    def _extract_chroma_features(self, audio_data: np.ndarray, sample_rate: int) -> list:
        """Extract chroma features"""
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio_data,
                sr=sample_rate,
                n_chroma=settings.N_CHROMA,
                n_fft=settings.N_FFT,
                hop_length=settings.HOP_LENGTH
            )
            
            # Statistical features for each chroma bin
            chroma_features = []
            for i in range(chroma.shape[0]):
                chroma_features.append(np.mean(chroma[i]))
            
            return chroma_features
            
        except Exception as e:
            logger.error(f"Chroma extraction error: {str(e)}")
            return [0.0] * settings.N_CHROMA
    
    def _extract_mel_features(self, audio_data: np.ndarray, sample_rate: int) -> list:
        """Extract mel spectrogram statistical features"""
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sample_rate,
                n_mels=settings.N_MEL,
                n_fft=settings.N_FFT,
                hop_length=settings.HOP_LENGTH
            )
            
            # Convert to log scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Statistical features
            mel_features = [
                np.mean(mel_spec_db),
                np.std(mel_spec_db),
                self._calculate_skewness(mel_spec_db.flatten()),
                self._calculate_kurtosis(mel_spec_db.flatten())
            ]
            
            return mel_features
            
        except Exception as e:
            logger.error(f"Mel spectrogram extraction error: {str(e)}")
            return [0.0] * 4
    
    def _extract_spectral_features(self, audio_data: np.ndarray, sample_rate: int) -> list:
        """Extract various spectral features"""
        try:
            spectral_features = []
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio_data, sr=sample_rate, hop_length=settings.HOP_LENGTH
            )[0]
            spectral_features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            
            # Spectral rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=sample_rate, hop_length=settings.HOP_LENGTH
            )[0]
            spectral_features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_data, sr=sample_rate, hop_length=settings.HOP_LENGTH
            )[0]
            spectral_features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio_data, hop_length=settings.HOP_LENGTH
            )[0]
            spectral_features.extend([np.mean(zcr), np.std(zcr)])
            
            # Root Mean Square Energy
            rms = librosa.feature.rms(
                y=audio_data, hop_length=settings.HOP_LENGTH
            )[0]
            spectral_features.extend([np.mean(rms), np.std(rms)])
            
            # Spectral contrast
            spectral_contrast = librosa.feature.spectral_contrast(
                y=audio_data, sr=sample_rate, hop_length=settings.HOP_LENGTH
            )
            spectral_features.extend([
                np.mean(spectral_contrast),
                np.std(spectral_contrast)
            ])
            
            # Tonnetz (Tonal centroid features)
            tonnetz = librosa.feature.tonnetz(
                y=audio_data, sr=sample_rate, hop_length=settings.HOP_LENGTH
            )
            spectral_features.extend([
                np.mean(tonnetz),
                np.std(tonnetz)
            ])
            
            # Pad or trim to ensure exactly 20 features
            while len(spectral_features) < 20:
                spectral_features.append(0.0)
            
            return spectral_features[:20]
            
        except Exception as e:
            logger.error(f"Spectral features extraction error: {str(e)}")
            return [0.0] * 20
    
    def _extract_rhythm_features(self, audio_data: np.ndarray, sample_rate: int) -> list:
        """Extract rhythm and tempo features"""
        try:
            rhythm_features = []
            
            # Tempo estimation
            tempo, beats = librosa.beat.beat_track(
                y=audio_data, sr=sample_rate, hop_length=settings.HOP_LENGTH
            )
            rhythm_features.append(tempo)
            
            # Beat strength
            onset_strength = librosa.onset.onset_strength(
                y=audio_data, sr=sample_rate, hop_length=settings.HOP_LENGTH
            )
            rhythm_features.extend([
                np.mean(onset_strength),
                np.std(onset_strength),
                np.max(onset_strength)
            ])
            
            # Rhythm regularity (coefficient of variation of beat intervals)
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                if len(beat_intervals) > 0:
                    rhythm_regularity = np.std(beat_intervals) / np.mean(beat_intervals)
                else:
                    rhythm_regularity = 0.0
            else:
                rhythm_regularity = 0.0
            
            rhythm_features.append(rhythm_regularity)
            
            # Ensure exactly 5 features
            while len(rhythm_features) < 5:
                rhythm_features.append(0.0)
            
            return rhythm_features[:5]
            
        except Exception as e:
            logger.error(f"Rhythm features extraction error: {str(e)}")
            return [0.0] * 5
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 3)
        except:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            return np.mean(((data - mean) / std) ** 4) - 3
        except:
            return 0.0
