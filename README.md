# 🎧 Bioacoustics Audio Classification

<p align="center">
  <img src="assets/frog-attack.gif" width="500">
</p>

# 🐸 Bioacoustics Audio Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Arduino](https://img.shields.io/badge/Arduino-Nano%2033%20BLE%20Sense-00979D?style=flat-square&logo=arduino&logoColor=white)
![Edge Impulse](https://img.shields.io/badge/Edge%20Impulse-TinyML-6236FF?style=flat-square)
![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9?style=flat-square)
![License](https://img.shields.io/badge/License-Research-lightgrey?style=flat-square)

**End-to-end bioacoustic monitoring pipeline — from raw `.wav` field recordings to real-time on-device frog species classification running entirely on a microcontroller.**

![Frog classification demo](assets/frog-attack.gif)

[Overview](#-project-overview) · [Architecture](#-system-architecture) · [Hardware](#-hardware-requirements) · [Setup](#%EF%B8%8F-installation--setup) · [Training](#-training-pipeline) · [Edge Deployment](#-edge-deployment) · [Project Structure](#-project-structure)

</div>

---

## 🌿 Project Overview

Frogs are highly sensitive ecological indicators — their calls reveal the health of wetlands, paddy fields, and forest ecosystems in ways that other sensors cannot. Yet continuous, large-scale acoustic monitoring has traditionally required trained field biologists and expensive equipment.

This project automates the full workflow end-to-end:

1. **Training (Python)** — A CNN-based classifier is trained on the **Mandookavani dataset** of Indian frog recordings. Audio is segmented, features are extracted (MFCCs / mel-spectrograms via `librosa`), and augmented with realistic field noise using `audiomentations`. The model is built in TensorFlow/Keras and exported to TFLite.

2. **Deployment (Arduino)** — The quantized model is packaged via Edge Impulse into an Arduino inference library and flashed onto an **Arduino Nano 33 BLE Sense**, which classifies frog calls in real time on-device — no cloud, no internet, no external compute.

The result is a palm-sized, battery-powered acoustic sensor you can zip-tie to a tree.

### Problems This Solves

| Challenge | Solution |
|-----------|----------|
| Expert-dependent manual surveys | Automated ML-based classification |
| Cloud inference needs connectivity | TFLite on-device inference via Edge Impulse |
| Long recordings are hard to label | Automated segmentation pipeline (`pydub`) |
| Noisy and variable field conditions | `audiomentations` augmentation during training |
| Single-sensor failure modes | Multi-sensor fusion sketch (mic + IMU + environment) |
| Continuous unattended monitoring | Streaming continuous inference sketch |

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════════╗
║                     TRAINING PIPELINE                                ║
║               Python 3.9 · TensorFlow 2.20 · uv                    ║
║                                                                      ║
║   Mandookavani_dataset/   (raw .wav recordings, by species)          ║
║           │                                                          ║
║           ▼                                                          ║
║   ┌──────────────────┐                                               ║
║   │  Segmentation    │  pydub — split long clips into N-sec windows  ║
║   └────────┬─────────┘                                               ║
║            │                                                         ║
║            ▼                                                         ║
║   ┌──────────────────┐                                               ║
║   │ Feature Extract  │  librosa — MFCC / Mel Spectrogram             ║
║   └────────┬─────────┘                                               ║
║            │                                                         ║
║            ▼                                                         ║
║   ┌──────────────────┐                                               ║
║   │  Augmentation    │  audiomentations — noise, pitch, time-stretch ║
║   └────────┬─────────┘                                               ║
║            │                                                         ║
║            ▼                                                         ║
║   ┌──────────────────┐                                               ║
║   │   CNN Training   │  TensorFlow / Keras                           ║
║   └────────┬─────────┘                                               ║
║            │                                                         ║
║            ▼                                                         ║
║   ┌───────────────────────┐                                          ║
║   │  TFLite Export        │  int8 post-training quantization         ║
║   │  + Edge Impulse Pack  │  → Arduino .zip inference library        ║
║   └───────────────────────┘                                          ║
╚══════════════════════════════════════════════════════════════════════╝
                        │
            Arduino inference library (.zip)
                        │
╔══════════════════════════════════════════════════════════════════════╗
║                    EDGE DEPLOYMENT                                   ║
║              Arduino Nano 33 BLE Sense · nRF52840                   ║
║                                                                      ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │ nano_ble33_sense_microphone.ino                             │    ║
║  │   Single audio window → DSP features → inference → Serial   │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │ nano_ble33_sense_microphone_continuous.ino                  │    ║
║  │   Continuous PDM stream → sliding window → rolling inference │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │ nano_ble33_sense_camera.ino                                 │    ║
║  │   OV7675 image frame → visual species identification        │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║  ┌─────────────────────────────────────────────────────────────┐    ║
║  │ nano_ble33_sense_fusion.ino                                 │    ║
║  │   Mic + IMU + temp/humidity → fused multi-sensor inference  │    ║
║  └─────────────────────────────────────────────────────────────┘    ║
║                          │                                           ║
║                          ▼                                           ║
║           Serial Monitor @ 115200 baud                              ║
║     Species label · Confidence score · DSP/inference timing         ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 🔧 Hardware Requirements

### Arduino Nano 33 BLE Sense *(required)*

| Component | Specification |
|-----------|---------------|
| MCU | Nordic nRF52840 — ARM Cortex-M4F @ 64 MHz |
| RAM | 256 KB SRAM |
| Flash | 1 MB |
| **Onboard mic** | **MP34DT05 PDM digital microphone** ← required for all audio sketches |
| IMU | LSM9DS1 — 9-axis (accel / gyro / magnetometer) |
| Environment | HTS221 (temp + humidity) · LPS22HB (pressure) |
| Gesture / light | APDS-9960 |
| Wireless | Bluetooth 5.0 BLE |
| Interface | Native USB — programming + Serial Monitor |

> ⚠️ **You need the Nano 33 BLE *Sense*** (not the plain Nano 33 BLE). Only the `Sense` variant carries the onboard PDM microphone.

### Optional Components

| Component | Required for |
|-----------|-------------|
| OV7675 camera module | `nano_ble33_sense_camera.ino` |
| Micro-USB cable | All sketches (programming + serial) |
| 3.7V LiPo battery | Untethered field deployment |
| Weatherproof enclosure | Outdoor / long-duration use |

### Computer Requirements (Training)

| | Minimum | Recommended |
|-|---------|-------------|
| OS | Ubuntu 20.04 / macOS 12 / Windows 10 | Ubuntu 22.04 LTS |
| RAM | 8 GB | 16 GB |
| GPU | CPU-only works | NVIDIA + CUDA (speeds training significantly) |
| Storage | 10 GB free | 20 GB+ |
| **Python** | **3.9** (pinned) | **3.9** |

---

## 💻 Software & Dependencies

This project uses **[`uv`](https://github.com/astral-sh/uv)** as its package manager — fast, modern, and reproducible. All dependencies are declared in `pyproject.toml` and fully locked in `uv.lock`. Python **3.9** is pinned via `.python-version`.

### Direct Dependencies (`pyproject.toml`)

| Package | Pinned version | Role |
|---------|----------------|------|
| `tensorflow` | ≥ 2.20.0 → resolves **2.20.0** on Py3.9 | Model training · TFLite export |
| `librosa` | ≥ 0.11.0 | Audio I/O · MFCC · mel-spectrogram |
| `audiomentations` | ≥ 0.42.0 → resolves **0.42.0** on Py3.9 | Data augmentation |
| `birdnetlib` | ≥ 0.18.0 → resolves **0.18.0** | Bioacoustic feature backbone |
| `sounddevice` | ≥ 0.5.5 | Live microphone recording |
| `soundfile` | ≥ 0.13.1 | `.wav` file read / write |
| `pydub` | ≥ 0.25.1 | Audio segmentation · format conversion |
| `numpy` | ≥ 2.0.2 → resolves **2.0.2** on Py3.9 | Array operations |
| `scipy` | ≥ 1.13.1 → resolves **1.13.1** on Py3.9 | Signal processing |
| `scikit-learn` | ≥ 1.6.1 → resolves **1.6.1** on Py3.9 | Metrics · train/test split |
| `matplotlib` | ≥ 3.9.4 → resolves **3.9.4** on Py3.9 | Plots · spectrograms |
| `tqdm` | ≥ 4.67.3 | Progress bars |

Key transitive dependencies resolved by `uv.lock`: `absl-py 2.3.1`, `audioread 3.1.0`, `astunparse 1.6.3`, `cffi 2.0.0`, `certifi 2026.2.25`, and the full TensorFlow runtime stack.

### Arduino IDE

| Component | Notes |
|-----------|-------|
| Arduino IDE 2.x | [arduino.cc/en/software](https://www.arduino.cc/en/software) |
| Board package | `Arduino Mbed OS Nano Boards` — install via Boards Manager |
| Inference library | Exported from Edge Impulse Studio as `.zip` |
| PDM library | Bundled with the Mbed board package |

---

## 📊 Dataset & Preprocessing

### Mandookavani Dataset

Raw recordings are stored under `Mandookavani_dataset/`, organised by species subdirectory. The dataset contains `.wav` field recordings of Indian frog species, originally provided for this research project.

```
Mandookavani_dataset/
├── species_A/
│   ├── rec_001.wav
│   └── ...
├── species_B/
├── species_C/
└── species_D/
```

### Audio Segmentation

Long field recordings are sliced into fixed-length clips using `pydub`, with silence filtering and amplitude normalisation. Processed clips land in `Training/final_data/`.

```
Raw field recording  (.wav, minutes long)
         │
         ▼  pydub
   Slice into N-second windows
         │
         ▼
   Drop silent / low-energy segments
         │
         ▼
   Normalise amplitude
         │
         ▼
   Training/final_data/<species>/<clip>.wav
```

### Feature Extraction (`librosa`)

Each clip is transformed into a 2-D feature map fed to the CNN:

```
Audio clip  (16 kHz · mono · .wav)
        │
        ├──▶  Mel Spectrogram
        │       n_mels = 40
        │       hop_length = 512
        │       → log-amplitude → normalised 2-D array
        │
        └──▶  MFCC
                n_mfcc = 13  (+delta features)
                → 2-D time-frequency array
```

See `Training/notebooks/` for parameter sweeps and visual comparisons.

### Data Augmentation (`audiomentations`)

Applied at training time to simulate real field recording variability:

| Transform | What it simulates |
|-----------|------------------|
| `AddGaussianNoise` | Environmental background noise / wind |
| `TimeStretch` | Speed variation without pitch change |
| `PitchShift` | Natural call variation across individual frogs |
| `Shift` | Random temporal offset within the clip window |
| `Gain` | Varying recording distance / microphone sensitivity |

---

## 🧠 Model Architecture

A **Convolutional Neural Network** built in TensorFlow / Keras takes the 2-D feature map (mel-spectrogram or MFCC) and outputs a per-class probability distribution. The architecture is kept compact to fit within the Nano 33 BLE Sense's 1 MB flash after quantisation.

```
Input: 2-D feature map (Mel Spectrogram or MFCC)
          │
          ▼
  ┌──────────────────────────────────────┐
  │  Conv2D → BatchNorm → ReLU          │   Low-level frequency patterns
  │  MaxPooling2D                        │
  └──────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────┐
  │  Conv2D → BatchNorm → ReLU          │   Temporal-spectral structures
  │  MaxPooling2D                        │
  └──────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────┐
  │  Conv2D → BatchNorm → ReLU          │   High-level call representations
  │  GlobalAveragePooling2D              │
  └──────────────────────────────────────┘
          │
          ▼
  ┌──────────────────────────────────────┐
  │  Dense → ReLU → Dropout             │   Classification head
  │  Dense → Softmax                    │
  └──────────────────────────────────────┘
          │
          ▼
  Output: probability per species class
```

### Export Path to Edge Hardware

```
Keras model (.h5)
     │
     ▼  tf.lite.TFLiteConverter
TFLite model (.tflite)  [float32]
     │
     ▼  Post-training int8 quantisation
Quantised model (~3–5× smaller, Nano 33 BLE Sense-compatible)
     │
     ▼  Edge Impulse Studio — Deploy → Arduino Library
Arduino .zip inference library
     │
     ▼  Arduino IDE — Sketch → Include Library → Add .ZIP
Flashed to Nano 33 BLE Sense via one of four .ino sketches
```

---

## 🛠️ Installation & Setup

### 1 — Clone the Repository

```bash
git clone https://github.com/NeoASJ/Bio-acoustics-.git
cd Bio-acoustics-
```

### 2 — Install `uv`

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3 — Sync the Environment

`uv` reads `.python-version` (→ Python 3.9), `pyproject.toml`, and `uv.lock` to build a fully reproducible virtual environment:

```bash
uv sync
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4 — Verify

```bash
python -c "import tensorflow, librosa, audiomentations, birdnetlib; print('All OK')"
```

### 5 — Arduino IDE

1. Download and install [Arduino IDE 2.x](https://www.arduino.cc/en/software)
2. Open **Tools → Board → Boards Manager** → search `Arduino Mbed OS Nano Boards` → Install
3. Connect the Nano 33 BLE Sense via USB
4. Select **Tools → Board → Arduino Nano 33 BLE Sense**
5. Select the correct **Port**

---

## 🎓 Training Pipeline

All training code and notebooks live in `Training/`. Outputs (weights, plots, logs) are written to `exps/`.

### Step 1 — Organise the Dataset

Place raw `.wav` recordings in `Mandookavani_dataset/`, one subdirectory per species. Then run segmentation:

```bash
python Training/segment_audio.py \
    --input_dir  Mandookavani_dataset/ \
    --output_dir Training/final_data/ \
    --duration   2.0 \
    --sr         16000
```

Or work interactively:

```bash
jupyter notebook Training/notebooks/
```

### Step 2 — Extract Features

```bash
python Training/extract_features.py \
    --data_dir    Training/final_data/ \
    --feature     mel_spectrogram \
    --n_mels      40 \
    --output_dir  Training/features/
```

### Step 3 — Train

```bash
python Training/train.py \
    --features_dir Training/features/ \
    --epochs       50 \
    --batch_size   32 \
    --output_dir   exps/run_001/
```

**Outputs in `exps/run_001/`:**

| File | Description |
|------|-------------|
| `model.h5` | Keras model weights |
| `model.tflite` | Quantised TFLite model |
| `labels.txt` | Integer → species name mapping |
| `training_history.png` | Loss / accuracy curves |
| `confusion_matrix.png` | Per-class classification breakdown |

### Step 4 — Sanity-Check the Entry Point

```bash
python main.py
# Hello from bio!
```

`main.py` is currently a minimal stub — extend it to wire your training or live-inference pipeline.

### Step 5 — Package for Edge Impulse

1. Sign into [Edge Impulse Studio](https://studio.edgeimpulse.com/) and create a project
2. Upload your processed audio clips (or connect to the Nano 33 BLE Sense data forwarder)
3. Design the **Impulse**: Audio → MFCC processing block → Neural Network classifier
4. Train and validate in Studio
5. **Deployment → Arduino Library → Build → Download `.zip`**

---

## 📡 Edge Deployment

The `edge-device/nano_ble33_sense/` folder holds four Arduino sketches. Install the Edge Impulse library first, then choose the sketch that matches your monitoring scenario.

### One-Time Library Install

```
Arduino IDE
  → Sketch → Include Library → Add .ZIP Library…
  → Select the .zip downloaded from Edge Impulse Studio
```

---

### `nano_ble33_sense_microphone.ino` — Single-Shot Inference

**What it does:** Waits 2 seconds, records one fixed-length audio window from the PDM microphone, runs DSP feature extraction and inference, prints the confidence scores — then repeats.

**Use when:** Verifying hardware + model integration; field spot-checks; demos.

```
Upload:
  Open edge-device/nano_ble33_sense/nano_ble33_sense_microphone/
       nano_ble33_sense_microphone.ino
  → Verify → Upload
  → Tools → Serial Monitor → 115200 baud
```

**Example output:**
```
Starting inferencing in 2 seconds...
Recording...
Recording done

Predictions (DSP: 118 ms., Classification: 9 ms., Anomaly: 0 ms.):
  Frog_A: 0.02344
  Frog_B: 0.87500   ← detected
  Frog_C: 0.07031
  Frog_D: 0.03125
```

---

### `nano_ble33_sense_microphone_continuous.ino` — Continuous Monitoring

**What it does:** Streams audio continuously from the PDM mic. Uses a sliding window (`EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW` slices) so inference runs on every new audio chunk without pausing. This is the primary sketch for real ecological monitoring.

**Use when:** Unattended long-duration field deployment; nightly frog activity logging.

```
Upload:
  Open edge-device/nano_ble33_sense/nano_ble33_sense_microphone_continuous/
       nano_ble33_sense_microphone_continuous.ino
  → Upload → Serial Monitor (115200 baud)
```

**Tip:** Add a confidence gate inside the sketch to suppress uncertain predictions:

```cpp
if (result.classification[ix].value > 0.80) {
    Serial.print(result.classification[ix].label);
    Serial.println(" DETECTED");
}
```

---

### `nano_ble33_sense_camera.ino` — Visual Identification

**What it does:** Captures frames from an attached OV7675 camera module, resizes and normalises them, then runs an image classifier to visually identify frog species in-frame. Outputs the predicted label and score over Serial.

**Use when:** Daytime camera-trap deployments; visual confirmation alongside acoustic detection; situations where calls are absent but frogs are visible.

**Requires:** OV7675 camera module physically connected to the Nano 33 BLE Sense camera header.

```
Upload:
  Open edge-device/nano_ble33_sense/nano_ble33_sense_camera/
       nano_ble33_sense_camera.ino
  → Upload → Serial Monitor (115200 baud)
```

---

### `nano_ble33_sense_fusion.ino` — Multi-Sensor Fusion

**What it does:** Simultaneously reads data from multiple onboard sensors, combines them into a single fused feature vector, and runs one unified inference pass per cycle.

**Fused sensor streams:**

| Sensor | Data | Ecological value |
|--------|------|-----------------|
| MP34DT05 PDM mic | Acoustic features (MFCC) | Primary call detection |
| LSM9DS1 IMU | Accelerometer + gyroscope | Physical disturbance near device |
| HTS221 | Temperature + humidity | Environmental context (frogs call more in warm, humid conditions) |

**Use when:** Reducing false positives in noisy environments; research requiring correlated multi-modal sensor data; advanced deployments where environmental context matters.

```
Upload:
  Open edge-device/nano_ble33_sense/nano_ble33_sense_fusion/
       nano_ble33_sense_fusion.ino
  → Upload → Serial Monitor (115200 baud)
```

---

### Reading the Serial Output (All Sketches)

```
Predictions (DSP: X ms., Classification: X ms., Anomaly: X ms.):
  <species_label>: <confidence>
  ...
```

| Field | Meaning |
|-------|---------|
| `DSP` | On-device feature extraction time (ms) |
| `Classification` | Neural network forward-pass time (ms) |
| `Anomaly` | Anomaly score — high = sound not matching any trained class |
| `confidence` | 0.0–1.0; highest value = predicted species |

**Confidence thresholds (suggested):**

| Confidence | Interpretation |
|-----------|---------------|
| > 0.80 | Strong detection — log the event |
| 0.50 – 0.80 | Uncertain — cross-check with time of day / humidity |
| < 0.50 | Likely background noise or unknown sound |

---

## 📁 Project Structure

```
Bio-acoustics-/
│
├── Mandookavani_dataset/                  # Raw .wav frog call recordings (by species)
│
├── Training/                              # Python training pipeline
│   ├── notebooks/                         # Jupyter notebooks: EDA, feature extraction,
│   │                                      #   model development, Edge Impulse workflow
│   └── final_data/                        # Cleaned, segmented clips (post-preprocessing)
│
├── edge-device/
│   └── nano_ble33_sense/                  # Arduino inference sketches
│       ├── nano_ble33_sense_microphone/
│       │   └── nano_ble33_sense_microphone.ino            # Single-shot audio inference
│       ├── nano_ble33_sense_microphone_continuous/
│       │   └── nano_ble33_sense_microphone_continuous.ino # Continuous streaming inference
│       ├── nano_ble33_sense_camera/
│       │   └── nano_ble33_sense_camera.ino                # Camera-based visual inference
│       └── nano_ble33_sense_fusion/
│           └── nano_ble33_sense_fusion.ino                # Multi-sensor fusion inference
│
├── exps/                                  # Experiment outputs: weights, metrics, plots
│
├── assets/                                # Media: GIFs and images used in this README
│   └── frog-attack.gif
│
├── main.py                                # Entry point stub ("Hello from bio!")
├── pyproject.toml                         # Project metadata + direct dependencies
├── uv.lock                                # Full locked dependency tree (2 698 lines)
├── .python-version                        # Pinned: 3.9
├── .gitignore
├── .gitattributes
└── README.md
```

---

## 🔬 Experiments (`exps/`)

Each training run writes its own subdirectory under `exps/`:

```
exps/
└── run_001/
    ├── model.h5                  # Keras weights
    ├── model.tflite              # Quantised TFLite model
    ├── labels.txt                # Class index → species name
    ├── training_history.png      # Loss / accuracy over epochs
    └── confusion_matrix.png      # Per-class accuracy breakdown
```

Notebooks in `Training/notebooks/` cover:
- Waveform and spectrogram visualisation of the Mandookavani dataset
- MFCC vs. mel-spectrogram feature comparison
- Hyperparameter sweeps (n_mels, hop_length, augmentation probabilities)
- Edge Impulse upload, impulse design, and export walkthrough

---

## 🗺️ Roadmap

- [ ] Ship complete CLI scripts for segmentation and feature extraction
- [ ] Add a `background` / `noise` class for out-of-distribution rejection
- [ ] Expand Mandookavani dataset with more species and recording environments
- [ ] BLE wireless logging — stream classification events to a mobile app
- [ ] Integrate `birdnetlib` embeddings as a transfer-learning backbone
- [ ] Power profiling and deep-sleep optimisation for multi-week battery life
- [ ] Web dashboard for aggregating events from multiple deployed sensors

---

## 🤝 Contributing

Contributions are especially welcome in these areas:
- Additional labelled frog call recordings for the Mandookavani dataset
- Improvements to the segmentation or augmentation pipeline
- Lighter model architectures (e.g. MobileNet-style depthwise convolutions)
- Field deployment case studies and accuracy reports

Open an issue or pull request on [GitHub](https://github.com/NeoASJ/Bio-acoustics-).

---

## 📄 License

This project is intended for **research and experimental purposes**. Refer to individual package licences for their respective terms. Dataset recordings may carry independent usage restrictions — verify before redistribution.

---

## 🙏 Acknowledgements

- **Mandookavani dataset** — Field recordings provided for this project
- [Edge Impulse](https://edgeimpulse.com/) — TinyML toolchain, DSP blocks, and Arduino library generation
- [librosa](https://librosa.org/) — Audio analysis and feature extraction
- [audiomentations](https://github.com/iver56/audiomentations) — Audio data augmentation library
- [birdnetlib](https://github.com/joeweiss/birdnetlib) — Bioacoustic analysis and BirdNET backbone
- [uv](https://github.com/astral-sh/uv) — Fast, reproducible Python package management
- The Arduino and TinyML communities for Nano 33 BLE Sense resources and examples

---

<div align="center">

Built for **ecological monitoring** · Powered by **TinyML** · Running on **edge hardware**

</div>
