# batch_test_api.py
import requests, os, json
from collections import Counter, defaultdict
from pathlib import Path

API = "http://127.0.0.1:8001/api/predict/file"
WAV_DIR = Path("test_wavs")  # put many test wavs here
LABEL_NEUTRAL = "neutral"

if not WAV_DIR.exists():
    print("Create folder test_wavs/ and add WAV files to test.")
    raise SystemExit(1)

counts = Counter()
conf_sums = defaultdict(float)
n = 0
failures = []

for p in WAV_DIR.glob("*.wav"):
    n += 1
    files = {"file": (p.name, open(p,"rb"), "audio/wav")}
    try:
        r = requests.post(API, files=files, data={"return_probabilities": "true"}, timeout=30)
        j = r.json()
        pred = j.get("predicted_emotion")
        conf = float(j.get("confidence", 0.0))
        probs = j.get("emotion_probabilities", {})
        counts[pred] += 1
        conf_sums[pred] += conf
        for lbl, v in probs.items():
            conf_sums[f"avg__{lbl}"] += float(v)
    except Exception as e:
        failures.append((p.name, str(e)))
        continue

print("Total tested:", n)
print("Counts:", counts)
print("Avg confidences per label (approx):")
for k in list(counts.keys()):
    print(k, "count:", counts[k], "avg_conf:", conf_sums[k]/counts[k] if counts[k] else 0.0)

if failures:
    print("Failures:", failures)
