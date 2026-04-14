# realtime_demo.py
import sounddevice as sd
import numpy as np
from classify_audio import FrogCallClassifier
import time 
classifier = FrogCallClassifier()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}")
    
    # Classify the audio chunk
    waveform = indata.flatten()
    predicted_class, confidence = classifier.classify(waveform)
    
    # Clear line and print result
    print(f"\rPrediction: {predicted_class} ({confidence:.2%})", end='')

print("Listening for frog calls... Press Ctrl+C to stop")
print("="*50)

try:
    with sd.InputStream(callback=audio_callback, 
                        samplerate=22050, 
                        channels=1,
                        blocksize=22050*3):  # 3 seconds blocks
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\n\nStopped listening")
