"""
Frog Species Classifier — Live Microphone
Uses BirdNET-Analyzer for zero-shot species identification from audio.

Install dependencies:
    pip install birdnetlib sounddevice soundfile numpy scipy

BirdNET covers 6000+ species including frogs/amphibians.
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


# ── Config ──────────────────────────────────────────────────────────────────

SAMPLE_RATE = 48000        # BirdNET expects 48kHz
RECORD_SECONDS = 3         # seconds per recording chunk
CHANNELS = 1               # mono
MIN_CONFIDENCE = 0.1       # minimum confidence to show result (0.0 - 1.0)

# Optional: set your location for better accuracy (or set to None)
LATITUDE = 12.9          # e.g. Karnataka, India
LONGITUDE = 74.8

# Frog-related keywords to filter results (set to None to show all species)
FROG_KEYWORDS = [
    "frog", "toad", "treefrog", "bullfrog", "rana", "bufo",
    "hyla", "microhyla", "raniceps", "cricket frog", "chorus frog"
]

# ── Analyzer setup ───────────────────────────────────────────────────────────

print("Loading BirdNET model... (first run may take a moment)")
analyzer = Analyzer()
print("Model ready.\n")


# ── Helpers ──────────────────────────────────────────────────────────────────

def record_chunk(duration=RECORD_SECONDS, sample_rate=SAMPLE_RATE):
    """Record audio from the default microphone."""
    print(f"  Recording {duration}s...", end=" ", flush=True)
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=CHANNELS,
        dtype="float32"
    )
    sd.wait()
    print("done.")
    return audio.flatten()


def save_temp_wav(audio, sample_rate=SAMPLE_RATE):
    """Save audio array to a temporary WAV file for BirdNET."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name


def is_frog_related(species_name):
    """Check if a detected species is frog/amphibian related."""
    if FROG_KEYWORDS is None:
        return True
    name_lower = species_name.lower()
    return any(kw in name_lower for kw in FROG_KEYWORDS)


def classify_audio(audio):
    """Run BirdNET on a numpy audio array, return detections."""
    wav_path = save_temp_wav(audio)
    try:
        recording = Recording(
            analyzer,
            wav_path,
            lat=LATITUDE,
            lon=LONGITUDE,
            min_conf=MIN_CONFIDENCE,
        )
        recording.analyze()
        return recording.detections
    finally:
        os.unlink(wav_path)


def display_results(detections, chunk_num):
    """Print frog detections in a readable format."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] Chunk #{chunk_num}")
    print("─" * 50)

    frog_hits = [d for d in detections if is_frog_related(d.get("common_name", ""))]

    if not frog_hits:
        all_hits = detections[:3]
        if all_hits:
            print("  No frog calls detected. Top detections:")
            for d in all_hits:
                conf = d.get("confidence", 0) * 100
                print(f"  · {d.get('common_name', '?')} ({conf:.1f}%)")
        else:
            print("  No detections above confidence threshold.")
    else:
        print(f"  Frog species detected ({len(frog_hits)} match{'es' if len(frog_hits) > 1 else ''}):")
        for d in sorted(frog_hits, key=lambda x: x.get("confidence", 0), reverse=True):
            conf = d.get("confidence", 0) * 100
            common = d.get("common_name", "Unknown")
            scientific = d.get("scientific_name", "")
            bar = "█" * int(conf / 10) + "░" * (10 - int(conf / 10))
            print(f"\n  {common}")
            print(f"  {scientific}")
            print(f"  [{bar}] {conf:.1f}%")

    print("─" * 50)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_classifier(num_chunks=None):
    """
    Run the live classifier.
    num_chunks: how many chunks to record (None = run forever until Ctrl+C)
    """
    print("=" * 50)
    print("  Frog Species Classifier — Live Microphone")
    print("=" * 50)
    print(f"  Sample rate : {SAMPLE_RATE} Hz")
    print(f"  Chunk length: {RECORD_SECONDS}s")
    print(f"  Location    : {LATITUDE}°N, {LONGITUDE}°E")
    print(f"  Min conf    : {MIN_CONFIDENCE * 100:.0f}%")
    print("  Press Ctrl+C to stop.\n")

    chunk_num = 0
    try:
        while num_chunks is None or chunk_num < num_chunks:
            chunk_num += 1
            audio = record_chunk()
            detections = classify_audio(audio)
            display_results(detections, chunk_num)
            time.sleep(0.1)  # brief pause between chunks

    except KeyboardInterrupt:
        print("\n\nStopped by user. Goodbye!")


if __name__ == "__main__":
    run_classifier()