import logging
logging.basicConfig(level=logging.INFO)

from services.audio_processor import AudioProcessor
from services.feature_extractor import FeatureExtractor

ap = AudioProcessor()
fe = FeatureExtractor()

audio_res = ap.load_audio_file("test_wavs/sine.webm")
if audio_res is None:
    print("load failed")
else:
    audio_data, sr = audio_res
    print(f"Loaded {len(audio_data)} samples at {sr} Hz")
    processed = ap.preprocess_audio(audio_data, sr)
    print(f"Preprocessed {len(processed)} samples")
    
    features = fe.extract_features(processed, sr)
    if features is None:
        print("Feature extraction returned None!")
    else:
        print(f"Extracted {len(features)} features")
