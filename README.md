<div align="center">

# 🐸 Bioacoustics Audio Classification

![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Arduino](https://img.shields.io/badge/Arduino-Nano%2033%20BLE%20Sense-00979D?style=flat-square&logo=arduino&logoColor=white)
![Edge Impulse](https://img.shields.io/badge/Edge%20Impulse-3D8F3D?style=flat-square&logo=arm&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-DE5FE9?style=flat-square)

**End-to-end bioacoustic monitoring pipeline — from PyTorch model training to TFLite deployment on Arduino Nano 33 BLE Sense**

![Frog classification demo](assets/frog-attack.gif)

[Overview](#-project-overview) · [Architecture](#-system-architecture) · [Hardware](#-hardware-requirements) · [Setup](#-installation--setup) · [Training](#-training-pipeline) · [Conversion](#-pytorch-to-tflite-conversion) · [Deployment](#-edge-deployment) · [Structure](#-project-structure)

</div>

---

## 🌿 Project Overview

Frogs are highly sensitive ecological indicators — their calls reveal the health of wetlands, paddy fields, and forest ecosystems in ways that other sensors cannot. Yet continuous, large-scale acoustic monitoring has traditionally required trained field biologists and expensive equipment.

This project automates the full workflow end-to-end:

1. **Training (PyTorch)** — A CNN-based classifier is trained on Indian frog recordings. Audio is segmented, features are extracted (mel-spectrograms via `librosa`). The model achieves **97.44% test accuracy**.

2. **Conversion** — PyTorch models (.pth) are converted through a robust pipeline: **PyTorch → ONNX → TensorFlow → TFLite (int8 quantization)** for edge deployment.

3. **Deployment (Arduino)** — The quantized model is packaged via Edge Impulse into an Arduino inference library and flashed onto an **Arduino Nano 33 BLE Sense**, which classifies frog calls in real time on-device — no cloud, no internet, no external compute.

The result is a palm-sized, battery-powered acoustic sensor you can zip-tie to a tree.

### Problems This Solves

| Challenge | Solution |
|-----------|----------|
| Expert-dependent manual surveys | ML-based classification |
| Cloud inference needs connectivity | TFLite on-device inference via Edge Impulse |
| PyTorch models can't run on MCUs | PyTorch → ONNX → TF → TFLite conversion pipeline |
| Long recordings are hard to label | Segmentation pipeline (`pydub`) |
| Noisy and variable field conditions | Data augmentation during training |
| Single-sensor failure modes | Multi-sensor fusion sketch (mic + IMU + environment) |
| Continuous unattended monitoring | Streaming continuous inference sketch |

---

## 🧠 Model Comparison

Two approaches were implemented and evaluated:

| Feature | **Custom CNN (Primary)** | **EfficientNet-B0 (Transfer Learning)** |
|:--------|:------------------------:|:----------------------------------------:|
| Framework | PyTorch | PyTorch |
| Approach | Built from scratch | Pretrained on ImageNet |
| **Test Accuracy** | **97.44%** | 70.00% |
| Training Time | Fast (~10 min) | Moderate (~30 min) |
| Model Size (FP32) | ~10 MB | ~20 MB |
| TFLite Size (int8) | ~2.5 MB | ~5 MB |
| Data Augmentation | ❌ | ✅ |
| Overfitting | Slight | Minimal |
| Inference Speed (MCU) | **~9 ms** | ~15 ms |
| **Recommendation** | ✅ **Primary model** | For larger datasets |

### Per-Class Performance Breakdown

**Custom CNN (97.44% overall):**
| Species | Accuracy |
|---------|----------|
| *Duttaphrynus melanostictus* | 96.43% |
| *Euphlyctis cyanophlyctis* | 95.83% |
| *Hoplobatrachus tigerinus* | 100.00% |
| *Microhyla ornata* | 97.50% |

**Transfer Learning (70.00% overall):**
| Species | Accuracy |
|---------|----------|
| *Duttaphrynus melanostictus* | 57.14% |
| *Euphlyctis cyanophlyctis* | 66.67% |
| *Hoplobatrachus tigerinus* | 100.00% |
| *Microhyla ornata* | 60.00% |

> **Conclusion:** The custom CNN significantly outperforms transfer learning for this specific 4-species classification task with the available dataset size.

---

## 🏗️ System Architecture
