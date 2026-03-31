# 🎧 Bioacoustics Audio Classification

<p align="center">
  <img src="assets/frog-attack.gif" width="500">
</p>

This repository focuses on building an **audio classification model using WAV files**...

This project focuses on building an **audio classification system for bioacoustic signals** using `.wav` recordings.

The goal is to detect and classify acoustic patterns—such as **frog calls and other environmental sounds**—using machine learning models designed to run efficiently on **edge devices**.

By combining **audio signal processing**, **machine learning**, and **embedded deployment**, the system aims to enable **real-time acoustic monitoring** for ecological and environmental applications.

This project not only trains an audio classification model but also deploys it as a **fully functional embedded AI system capable of real-time bioacoustic inference on microcontroller hardware**.

The overall workflow of the project includes:

* 🎧 **Audio Data Collection** – Raw WAV recordings of environmental sounds
* ✂️ **Audio Segmentation** – Splitting recordings into meaningful segments
* 📊 **Feature Extraction** – Generating features such as spectrograms or MFCCs
* 🤖 **Model Training** – Training classification models using Python frameworks
* ⚡ **Edge Optimization** – Preparing the model for deployment on low-power hardware
* 📡 **Real-time Inference** – Detecting frog calls or acoustic events directly on edge devices

---

# Project Structure

```
Bioacoustics/
│
├── Mandookavani_dataset/
│   └── Raw dataset provided by mahesh
│
├── segmentation/
│   ├── final_data/
│   │   └── Processed dataset used for training
│   │
│   └── notebooks/
│       └── Jupyter notebooks for preprocessing, experimentation, and model development
│
└── README.md
```

---

# Dataset Overview

### Raw Dataset

```
Mandookavani_dataset/
```

This directory contains the **original WAV recordings** provided for the project.
These recordings may include long audio clips, background noise, and multiple acoustic events.

---

### Processed Dataset

```
segmentation/final_data/
```

This folder contains **segmented and cleaned audio samples** prepared for model training.

Typical preprocessing steps include:

* Audio segmentation
* Noise filtering
* Signal normalization
* Preparing audio segments for feature extraction

---

# Experiments and Development

```
segmentation/notebooks/
```

This directory contains **Jupyter notebooks used for experimentation and development**, including:

* Audio waveform visualization
* Spectrogram generation
* Feature extraction (MFCC / Mel Spectrogram)
* Model training experiments
* Evaluation and analysis

---

# Audio Processing Pipeline

```
WAV Audio
   │
   ▼
Preprocessing
   │
   ▼
Audio Segmentation
   │
   ▼
Feature Extraction
(MFCC / Spectrogram)
   │
   ▼
Model Training
   │
   ▼
Model Optimization
   │
   ▼
Edge Deployment
```

---

# Technologies Used

| Technology           | Purpose                            |
| -------------------- | ---------------------------------- |
| Python               | Data processing and model training |
| Jupyter Notebook     | Experimentation and visualization  |
| NumPy / SciPy        | Audio signal processing            |
| PyTorch / TensorFlow | Deep learning model development    |
| C                    | Edge device inference runtime      |
| Edge AI frameworks   | Model deployment                   |

---

# Edge Deployment

The final goal of this project is to run the trained model on **low-power edge devices**, enabling real-time acoustic monitoring.

Benefits include:

* Low-latency inference
* Reduced power consumption
* Autonomous wildlife monitoring
* Continuous environmental sensing

Possible deployment platforms include:

* Microcontrollers
* Embedded Linux devices
* Edge AI modules
* Custom acoustic sensor hardware

---

# 🧩 Hardware Implementation

The system is deployed on a microcontroller to perform **real-time acoustic inference directly on-device**, without requiring internet connectivity.

## 🔌 Hardware Used

- Arduino Nano 33 BLE Sense  
  - Built-in PDM microphone  
  - ARM Cortex-M processor  
  - Optimized for TinyML and edge AI applications  

---

## ⚙️ Firmware Overview

The trained model is converted into an **Arduino-compatible inference library** and flashed onto the microcontroller.

The firmware performs:

1. 🎤 Audio capture using onboard microphone  
2. 🧠 Buffering of real-time audio samples  
3. 🤖 Running inference using embedded ML model  
4. 📊 Outputting confidence scores for each species  

---

### 📟 Example Output

```text
Predictions:
Frog_A: 0.12
Frog_B: 0.87  <-- highest confidence
Frog_C: 0.01
```

---

## 🔁 Real-Time Inference Flow
```
Microphone Input
│
▼
PDM Audio Buffer
│
▼
Feature Extraction (DSP)
│
▼
Embedded Model (TFLite)
│
▼
Confidence Scores Output
```
---
# 💻 Embedded Code Highlights

The embedded firmware is built using Edge AI inference libraries and Arduino framework.

## 🔑 Key Functionalities

### 🎤 Audio Sampling
```cpp
PDM.begin(1, EI_CLASSIFIER_FREQUENCY);

# Getting Started

Clone the repository:

```
git clone https://github.com/NeoASJ/Bio-acoustics-.git
```

Navigate into the project directory:

```
cd Bioacoustics
```

---

# Future Improvements

Planned improvements include:

* Automated audio preprocessing pipeline
* Real-time audio stream classification
* Edge hardware optimization
* Dataset augmentation and labeling improvements

---

# Contributors

Bioacoustics Research Project

Developed for building **efficient bioacoustic monitoring systems using machine learning and edge computing**.

---

# License

This project is intended for **research and experimental purposes**.
