import numpy as np
import scipy.io.wavfile as wav
import requests
import json
import os

API_URL = "http://127.0.0.1:3000/api/predict/file"
OUTPUT_DIR = "test_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_tone(filename, freq=440.0, duration=3.0, sample_rate=22050):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # Generate a pure sine wave, normalize to 16-bit range
    tone = (np.sin(freq * t * 2 * np.pi) * 32767).astype(np.int16)
    filepath = os.path.join(OUTPUT_DIR, filename)
    wav.write(filepath, sample_rate, tone)
    return filepath

def generate_noise(filename, duration=3.0, sample_rate=22050):
    # Generate white noise
    noise = (np.random.normal(0, 1, int(sample_rate * duration)) * 10000).astype(np.int16)
    filepath = os.path.join(OUTPUT_DIR, filename)
    wav.write(filepath, sample_rate, noise)
    return filepath

def test_api(filepath):
    print(f"\nTesting {filepath}...")
    with open(filepath, 'rb') as f:
        files = {'file': (os.path.basename(filepath), f, 'audio/wav')}
        data = {'return_probabilities': 'true'}
        try:
            response = requests.post(API_URL, files=files, data=data)
            if response.status_code == 200:
                result = response.json()
                print(f"Predicted Emotion: {result.get('predicted_emotion')}")
                print(f"Confidence: {result.get('confidence'):.2%}")
                print("Probabilities:")
                for em, prob in result.get('emotion_probabilities', {}).items():
                    print(f"  {em}: {prob:.2%}")
            else:
                print(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    print("Generating sample audio files...")
    
    tone_file = generate_tone("sine_wave_440hz.wav", freq=440.0)
    noise_file = generate_noise("white_noise.wav")
    
    print("Sending to Emotion Recognition API...")
    test_api(tone_file)
    test_api(noise_file)
