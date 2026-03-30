"""
Frog Species Classifier — Live Microphone + File Mode
Uses BirdNET-Analyzer for zero-shot species identification from audio.

Install:
    pip install birdnetlib sounddevice soundfile numpy scipy tensorflow

HOW TO USE IN JUPYTER:
    # Test with a file:
    classify_file(r"C:\path\to\frog.wav")

    # Live mic:
    run_live()
"""

import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import time
from datetime import datetime
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer


# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 48000
RECORD_SECONDS = 5
CHANNELS       = 1
MIN_CONFIDENCE = 0.01    # very low for testing — raise to 0.1 in production

LATITUDE  = 12.9
LONGITUDE = 74.8

FROG_KEYWORDS = [
    "frog", "toad", "treefrog", "bullfrog", "rana", "bufo",
    "hyla", "microhyla", "cricket frog", "chorus frog",
    "nyctibatrachus", "indirana", "sphaerotheca", "fejervarya",
]

# ── Analyzer setup ────────────────────────────────────────────────────────────

print("Loading BirdNET model...")
analyzer = Analyzer()
print("Model ready.\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def record_chunk(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    print(f"  Recording {duration}s...", end=" ", flush=True)
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=CHANNELS, dtype="float32")
    sd.wait()
    print("done.")
    return audio.flatten()


def save_temp_wav(audio, sample_rate=SAMPLE_RATE):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name


def is_frog(name):
    return any(kw in name.lower() for kw in FROG_KEYWORDS)


def run_birdnet(path):
    recording = Recording(analyzer, path, lat=LATITUDE, lon=LONGITUDE,
                          min_conf=MIN_CONFIDENCE)
    recording.analyze()
    return recording.detections


def display_results(detections, label):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] {label}")
    print("-" * 52)

    if not detections:
        print("  No sounds detected — mic may be silent.")
        print("  Tip: play a frog call near your mic to test.")
        print("-" * 52)
        return

    frog_hits  = [d for d in detections if is_frog(d.get("common_name", ""))]
    other_hits = [d for d in detections if not is_frog(d.get("common_name", ""))]

    if frog_hits:
        print(f"  FROG DETECTED — {len(frog_hits)} species:")
        for d in sorted(frog_hits, key=lambda x: x.get("confidence", 0), reverse=True):
            conf   = d.get("confidence", 0) * 100
            bar    = "#" * int(conf / 10) + "." * (10 - int(conf / 10))
            print(f"\n  *** {d.get('common_name', '?')}")
            print(f"      {d.get('scientific_name', '')}")
            print(f"      [{bar}] {conf:.1f}%")
    else:
        print("  No frog calls detected.")

    if other_hits:
        print(f"\n  Other sounds heard:")
        for d in sorted(other_hits, key=lambda x: x.get("confidence", 0), reverse=True)[:3]:
            print(f"  . {d.get('common_name', '?')} ({d.get('confidence', 0)*100:.1f}%)")

    print("-" * 52)


# ── Public API ────────────────────────────────────────────────────────────────

def classify_file(filepath):
    """
    Classify a WAV/MP3 file. Use this in Jupyter to test.
    Example:
        classify_file(r"C:\\Users\\HP\\Downloads\\bullfrog.wav")
    """
    print(f"Classifying: {filepath}\n")
    detections = run_birdnet(filepath)
    display_results(detections, os.path.basename(filepath))


def run_live():
    """
    Start live microphone classification. Press Ctrl+C (or interrupt kernel) to stop.
    """
    print("=" * 52)
    print("  Frog Classifier — Live Mic")
    print("=" * 52)
    print(f"  Chunk: {RECORD_SECONDS}s | Min conf: {MIN_CONFIDENCE*100:.0f}%")
    print(f"  Location: {LATITUDE}N, {LONGITUDE}E")
    print("  Interrupt kernel to stop.\n")

    chunk_num = 0
    try:
        while True:
            chunk_num += 1
            audio    = record_chunk()
            wav_path = save_temp_wav(audio)
            try:
                detections = run_birdnet(wav_path)
            finally:
                os.unlink(wav_path)
            display_results(detections, f"Chunk #{chunk_num}")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nStopped.")


# ── Script mode (run directly, not in Jupyter) ────────────────────────────────

if __name__ == "__main__":
    import sys
    # Only use sys.argv when run as a script — not in Jupyter
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        classify_file(sys.argv[1])
    else:
        run_live()
    