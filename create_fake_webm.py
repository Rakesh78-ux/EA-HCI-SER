import subprocess
import imageio_ffmpeg
import os

wav_file = "test_wavs/sine_wave_440hz.wav"
webm_file = "test_wavs/sine.webm"

if os.path.exists(webm_file):
    os.remove(webm_file)

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
subprocess.run([ffmpeg_exe, "-i", wav_file, "-c:a", "libopus", webm_file], check=True)
print(f"Created {webm_file}")
