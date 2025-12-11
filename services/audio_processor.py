import numpy as np
import librosa
import logging
from typing import Optional, Tuple
from pydub import AudioSegment
import io
import tempfile
import os

from config import settings

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.initialized = False

    def initialize(self):
        """Initialize the audio processor"""
        self.initialized = True
        logger.info("Audio processor initialized")

    def load_audio_file(self, file_path: str) -> Optional[Tuple[np.ndarray, int]]:
        """Load audio file and return audio data with sample rate"""
        try:
            # Load audio using librosa
            audio_data, sample_rate = librosa.load(
                file_path,
                sr=None,  # Keep original sample rate initially
                mono=True
            )

            logger.info(f"Loaded audio file: {file_path}, shape: {audio_data.shape}, sr: {sample_rate}")
            return audio_data, sample_rate

        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {str(e)}")
            return None

    def load_audio_from_bytes(self, audio_bytes: bytes, file_format: str = "wav") -> Optional[Tuple[np.ndarray, int]]:
        """Load audio from bytes data"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=f".{file_format}", delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name

            try:
                # Load from temporary file
                audio_data, sample_rate = self.load_audio_file(temp_file_path)
                return audio_data, sample_rate

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        except Exception as e:
            logger.error(f"Error loading audio from bytes: {str(e)}")
            return None

    def convert_audio_format(self, input_bytes: bytes, input_format: str, output_format: str = "wav") -> Optional[bytes]:
        """Convert audio from one format to another using pydub"""
        try:
            # Load audio with pydub
            audio_segment = AudioSegment.from_file(
                io.BytesIO(input_bytes),
                format=input_format
            )

            # Convert to target format
            output_buffer = io.BytesIO()
            audio_segment.export(output_buffer, format=output_format)

            return output_buffer.getvalue()

        except Exception as e:
            logger.error(f"Error converting audio format: {str(e)}")
            return None

    def preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Preprocess audio data for emotion recognition (with robust dtype handling and resample)"""
        # DEBUG LOG — preprocess input stats
        logger.info(
            "AUDIO_PROCESSOR.preprocess_audio: input len=%d sample_rate=%s dtype=%s",
            len(audio_data) if hasattr(audio_data, '__len__') else -1,
            sample_rate,
            getattr(audio_data, "dtype", "unknown")
        )
        try:
            logger.info(
                "AUDIO_PROCESSOR.preprocess_audio: input min/max/mean = %.6f / %.6f / %.6f",
                float(np.min(audio_data)),
                float(np.max(audio_data)),
                float(np.mean(audio_data))
            )
        except Exception:
            logger.exception("Failed to log audio stats in preprocess_audio")

        # ===== TOLERANT TYPE CONVERSION =====
        # Accept common incoming encodings (int16, uint8, float32) and convert to float32 in [-1, 1]
        try:
            audio_data = np.asarray(audio_data)
            if audio_data.dtype == np.int16:
                audio_data = (audio_data.astype("float32") / 32768.0)
                logger.info("Converted int16 -> float32")
            elif audio_data.dtype == np.uint8:
                audio_data = ((audio_data.astype("float32") - 128.0) / 128.0)
                logger.info("Converted uint8 -> float32")
            else:
                audio_data = audio_data.astype("float32")
        except Exception:
            logger.exception("Failed to normalize audio dtype")

        # ===== RESAMPLE TO TARGET SAMPLE RATE =====
        try:
            target_sr = getattr(settings, "SAMPLE_RATE", None)
            if target_sr is not None and sample_rate != target_sr:
                audio_data = self.resample_audio(audio_data, orig_sr=sample_rate, target_sr=target_sr)
                sample_rate = target_sr
        except Exception:
            logger.exception("Resampling failed")

        try:
            # 1. Normalize audio amplitude to [-1,1]
            audio_data = self.normalize_audio(audio_data)

            # 2. Remove silence
            audio_data = self.trim_silence(audio_data)

            # 3. Apply pre-emphasis filter
            audio_data = self.apply_preemphasis(audio_data)

            # 4. Ensure minimum length
            audio_data = self.ensure_minimum_length(audio_data, sample_rate)

            # 5. Limit maximum length to avoid memory issues
            audio_data = self.limit_maximum_length(audio_data, sample_rate)

            return audio_data

        except Exception as e:
            logger.error(f"Error preprocessing audio: {str(e)}")
            return audio_data

    def normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        try:
            # Remove DC offset
            audio_data = audio_data - np.mean(audio_data)

            # Normalize to [-1, 1] range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val

            return audio_data

        except Exception as e:
            logger.error(f"Error normalizing audio: {str(e)}")
            return audio_data

    def trim_silence(self, audio_data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Remove silence from beginning and end of audio"""
        try:
            # Find non-silent regions
            non_silent = np.abs(audio_data) > threshold

            if np.any(non_silent):
                # Find first and last non-silent samples
                first_nonzero = np.argmax(non_silent)
                last_nonzero = len(audio_data) - np.argmax(non_silent[::-1]) - 1

                # Trim audio
                audio_data = audio_data[first_nonzero:last_nonzero + 1]

            return audio_data

        except Exception as e:
            logger.error(f"Error trimming silence: {str(e)}")
            return audio_data

    def apply_preemphasis(self, audio_data: np.ndarray, coeff: float = 0.97) -> np.ndarray:
        """Apply pre-emphasis filter to enhance high frequencies"""
        try:
            if len(audio_data) > 1:
                emphasized = np.append(audio_data[0], audio_data[1:] - coeff * audio_data[:-1])
                return emphasized
            return audio_data

        except Exception as e:
            logger.error(f"Error applying pre-emphasis: {str(e)}")
            return audio_data

    def ensure_minimum_length(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Ensure audio meets minimum length requirement"""
        try:
            min_samples = int(settings.DURATION * 0.5 * sample_rate)  # 0.5 * target duration

            if len(audio_data) < min_samples:
                # Pad with zeros or repeat audio
                if len(audio_data) > 0:
                    # Repeat audio to meet minimum length
                    repeats = int(np.ceil(min_samples / len(audio_data)))
                    audio_data = np.tile(audio_data, repeats)[:min_samples]
                else:
                    # Create silent audio if input is empty
                    audio_data = np.zeros(min_samples)

            return audio_data

        except Exception as e:
            logger.error(f"Error ensuring minimum length: {str(e)}")
            return audio_data

    def limit_maximum_length(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Limit audio to maximum length to avoid processing issues"""
        try:
            max_samples = int(settings.DURATION * 2 * sample_rate)  # 2x target duration

            if len(audio_data) > max_samples:
                # Take the middle portion of the audio
                start_idx = (len(audio_data) - max_samples) // 2
                audio_data = audio_data[start_idx:start_idx + max_samples]

            return audio_data

        except Exception as e:
            logger.error(f"Error limiting maximum length: {str(e)}")
            return audio_data

    def resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        try:
            if orig_sr != target_sr:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=orig_sr,
                    target_sr=target_sr
                )
                logger.info(f"Audio resampled from {orig_sr}Hz to {target_sr}Hz")

            return audio_data

        except Exception as e:
            logger.error(f"Error resampling audio: {str(e)}")
            return audio_data

    def segment_audio(self, audio_data: np.ndarray, sample_rate: int, segment_duration: float = 3.0) -> list:
        """Segment audio into fixed-length chunks"""
        try:
            segment_samples = int(segment_duration * sample_rate)
            segments = []

            # Create overlapping segments
            hop_samples = segment_samples // 2  # 50% overlap

            for start in range(0, len(audio_data) - segment_samples + 1, hop_samples):
                end = start + segment_samples
                segment = audio_data[start:end]
                segments.append(segment)

            # Handle remaining audio if any
            if len(audio_data) > segment_samples and len(segments) > 0:
                last_segment_start = len(audio_data) - segment_samples
                if last_segment_start > segments[-1].shape[0]:  # Avoid duplicate
                    last_segment = audio_data[last_segment_start:]
                    if len(last_segment) == segment_samples:
                        segments.append(last_segment)

            return segments

        except Exception as e:
            logger.error(f"Error segmenting audio: {str(e)}")
            return [audio_data]

    def validate_audio_file(self, file_path: str) -> bool:
        """Validate if file is a supported audio format"""
        try:
            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext not in settings.ALLOWED_EXTENSIONS:
                return False

            # Try to load a small portion to verify format
            try:
                librosa.load(file_path, sr=None, duration=0.1)
                return True
            except:
                return False

        except Exception as e:
            logger.error(f"Error validating audio file: {str(e)}")
            return False