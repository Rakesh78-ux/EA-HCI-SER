import requests
import json
import base64

API_URL = "http://localhost:3000/api/predict/file"

# Generate a dummy tiny valid webm if possible, or just send a minimal valid file renamed as webm
# A 1-second sine wave in webm format would be better, but we can just use the wav file renamed.
# Actually, the API checks librosa.load(). librosa.load() uses soundfile or audioread.
# Let's send a fake webm file!

# For absolute certainty, let's write a tiny script to save the wav as an actual upload parameter:
import shutil
import io

with open("test_wavs/sine.webm", "rb") as f:
    webm_bytes = f.read()

# Send the real webm bytes
print("Sending actual .webm bytes...")
files = {"file": ("recording.webm", webm_bytes, "audio/webm")}
data = {"return_probabilities": "true", "transcript": "test"}

resp = requests.post(API_URL, files=files, data=data)
print(f"Status: {resp.status_code}")
print(f"Response: {resp.json()}")

