# diagnose_model.py
import sys
import traceback
import json
import numpy as np
import soundfile as sf
from pathlib import Path
import importlib
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("diagnose")

# config import
from config import settings

def print_header(s):
    print("\n" + "="*10 + " " + s + " " + "="*10)

print_header("ENV")
try:
    import sklearn
    print("python executable:", sys.executable)
    print("sklearn version:", sklearn.__version__)
except Exception as e:
    print("could not import sklearn:", e)

print_header("MODEL FILES")
print("MODEL_PATH:", settings.MODEL_PATH, "exists?", Path(settings.MODEL_PATH).exists())
print("SCALER_PATH:", settings.SCALER_PATH, "exists?", Path(settings.SCALER_PATH).exists())

print_header("LOAD CLASSIFIER")
try:
    from models.emotion_classifier import EmotionClassifier
    clf = EmotionClassifier()
    # capture stdout/logging of load
    clf.load_model()
    print("classifier.is_trained:", clf.is_trained)
    print("model type:", type(clf.model))
    print("model.classes_ (if any):", getattr(clf.model, "classes_", None))
    print("scaler type:", type(clf.scaler))
except Exception:
    print("Failed to instantiate/load classifier:")
    traceback.print_exc()

print_header("TEST AUDIO (create tone if none exists)")
test_wav = Path("test_samples/example.wav")
if not test_wav.exists():
    print("No test WAV found. Creating synthetic tone at", test_wav)
    test_wav.parent.mkdir(parents=True, exist_ok=True)
    sr = getattr(settings, "SAMPLE_RATE", 16000)
    dur = min(getattr(settings, "DURATION", 2.0), 2.0)
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    tone = 0.2 * np.sin(2*np.pi*220*t).astype(np.float32)
    sf.write(str(test_wav), tone, sr)
else:
    print("Found test WAV:", test_wav)

print_header("FEATURE EXTRACTION")
try:
    # load audio and preprocess like pipeline
    data, sr = sf.read(str(test_wav))
    if data.ndim > 1:
        data = data.mean(axis=1)
    print("Loaded test wav sr=", sr, "samples=", len(data))
    # import feature extractor
    from services.feature_extractor import FeatureExtractor
    fe = FeatureExtractor()
    feat = fe.extract_features(data.astype(np.float32), sr)
    print("feature output type:", type(feat), "shape:", None if feat is None else getattr(feat,'shape',None))
    if feat is not None:
        print("feature min/max/mean:", float(np.min(feat)), float(np.max(feat)), float(np.mean(feat)))
except Exception:
    print("Feature extraction failed:")
    traceback.print_exc()

print_header("DIRECT PREDICTION (no HTTP) — will show full returned dict")
try:
    # call predict and print everything
    pred = clf.predict((data.astype(np.float32)), sr)
    print("PREDICT returned:")
    print(json.dumps(pred, indent=2))
    # check uniform
    vals = list(pred.get("emotions", {}).values())
    if vals:
        uniform = all(abs(v - vals[0]) < 1e-9 for v in vals)
        print("Uniform distribution?", uniform, "first_value=", vals[0])
    else:
        print("No emotion probabilities returned")
except Exception:
    print("Prediction exception:")
    traceback.print_exc()

print_header("TRY LOADING PICKLED MODEL WITH JOBLIB (sanity)")
try:
    import joblib
    p = Path(settings.MODEL_PATH)
    if p.exists():
        m = joblib.load(str(p))
        print("joblib load type:", type(m), "classes_:", getattr(m, "classes_", None))
    else:
        print("Model path not present, skipping joblib load")
except Exception:
    print("joblib load failed:")
    traceback.print_exc()

print_header("FIN")
print("Done diagnostics.")
