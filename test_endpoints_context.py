import requests
import os

API_URL = "http://localhost:3000/api/predict/file"

# We assume test_wavs/sine_wave_440hz.wav exists from earlier steps
file_path = "test_wavs/sine_wave_440hz.wav"

def test_prediction(transcript=None):
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"return_probabilities": "true"}
        if transcript is not None:
            data["transcript"] = transcript

        print(f"\nTesting with transcript: '{transcript}'")
        try:
            resp = requests.post(API_URL, files=files, data=data)
            if resp.status_code == 200:
                result = resp.json()
                print(f"Status codes: OK. Emotion: {result['predicted_emotion']} ({result['confidence']:.2f})")
                
                print("Probabilities:")
                emotions = result.get('emotion_probabilities', {})
                for em, prob in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {em}: {prob:.2f}")
            else:
                print(f"Failed. Status: {resp.status_code}, {resp.text}")
        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    if not os.path.exists(file_path):
        print(f"Cannot find {file_path}")
    else:
        test_prediction(None) # baseline
        test_prediction("i hate you so much") # should boost angry
        test_prediction("this is wonderful and awesome") # should boost happy
