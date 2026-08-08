# AI-Based Road Damage Detection for Autonomous Systems

### Comparative Evaluation of RT-DETR-L and YOLOv8-L on the RDD2022 Dataset with Edge AI Deployment on NVIDIA Jetson Nano

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![RT-DETR](https://img.shields.io/badge/RT--DETR-L-success)
![YOLOv8](https://img.shields.io/badge/YOLOv8-L-yellow)
![Jetson Nano](https://img.shields.io/badge/NVIDIA-Jetson%20Nano-green)
![OAK-D](https://img.shields.io/badge/Luxonis-OAK--D-blueviolet)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange)

---

# Abstract

This project presents the design, implementation, evaluation, and embedded deployment of an AI-based road damage detection system for autonomous and intelligent transportation applications.

The work compares two state-of-the-art object detection models:

- RT-DETR-L
- YOLOv8-L

using the RDD2022 dataset under identical training conditions and evaluation metrics.

The project extends beyond model comparison by deploying the vision pipeline onto an NVIDIA Jetson Nano with a Luxonis OAK-D smart camera, demonstrating a complete edge AI solution suitable for robotics and autonomous systems.

---

# Motivation

Road surface damage directly affects transportation safety, driving comfort, and infrastructure maintenance costs.

Traditional manual inspection is expensive, time-consuming, and difficult to scale.

Recent advances in deep learning and edge AI enable automatic road inspection using onboard cameras and embedded computers.

The objective of this project is to investigate whether modern real-time object detectors can accurately identify different categories of road damage while remaining suitable for deployment on resource-constrained embedded platforms.

---

# Project Objectives

The main objectives of this project are:

- Develop a reproducible road damage detection pipeline
- Compare RT-DETR-L and YOLOv8-L under identical experimental conditions
- Evaluate both models using standard object detection metrics
- Analyze strengths and weaknesses of transformer-based and one-stage detectors
- Deploy the detection pipeline on NVIDIA Jetson Nano
- Integrate Luxonis OAK-D for real-time embedded vision
- Build a foundation for future autonomous mobile robot applications

---

# System Overview

```
                Road Image
                     │
                     ▼
             Luxonis OAK-D Camera
                     │
                     ▼
             NVIDIA Jetson Nano
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
      RT-DETR-L              YOLOv8-L
         │                       │
         └───────────┬───────────┘
                     ▼
           Road Damage Detection
                     │
                     ▼
          Visualization & Evaluation
```

---

# Hardware Platform

The embedded AI system consists of:

- NVIDIA Jetson Nano
- Luxonis OAK-D Camera
- USB Wi-Fi Adapter
- Raspberry Pi Pico (robot controller)
- ESP32 Motor Controller
- Four-wheel robotic platform
- DualShock 4 wireless controller
- External battery power system

---

# Software Stack

Development and deployment use:

- Ubuntu Linux
- Python
- PyTorch
- OpenCV
- Ultralytics
- RT-DETR
- DepthAI SDK
- Git
- GitHub

---

# Dataset

Dataset:

**RDD2022 (Road Damage Detection Dataset)**

Road damage classes:

- D00
- D10
- D20
- D40
- Other

---

# AI Models

## RT-DETR-L

RT-DETR-L is a transformer-based end-to-end object detector that eliminates Non-Maximum Suppression (NMS) and performs object prediction directly through transformer decoding.

Advantages:

- End-to-end detection
- Strong localization
- Modern transformer architecture

---

## YOLOv8-L

YOLOv8-L is a one-stage object detector designed for high-speed inference while maintaining strong detection accuracy.

Advantages:

- Fast inference
- Excellent real-time performance
- Mature deployment ecosystem

---

# Evaluation Metrics

The models are evaluated using:

- Precision
- Recall
- mAP50
- mAP50-95

---

# Experimental Results

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---------|----------|--------|---------|------------|
| RT-DETR-L | 0.696 | 0.602 | 0.643 | 0.331 |
| YOLOv8-L | 0.670 | 0.603 | 0.643 | 0.357 |

---

# Key Findings

- Both models achieved nearly identical mAP50.
- RT-DETR-L achieved slightly higher Precision.
- YOLOv8-L achieved higher mAP50-95.
- YOLOv8-L demonstrated better localization quality under the current experimental setup.

---

# Current Project Status

## Dataset

- ✅ Dataset prepared

## Training

- ✅ YOLOv8-L completed

- ✅ RT-DETR-L completed

## Evaluation

- ✅ Model comparison completed

## Edge AI Deployment

- ✅ Jetson Nano configured

- ✅ DepthAI installed

- ✅ OAK-D camera detected

- 🔄 Real-time camera pipeline

- 🔄 Edge inference optimization

---

# Repository Structure

```
road-damage-detection-rtdetr-vs-yolov8/

│

├── README.md

├── docs/

├── data/

├── models/

├── notebooks/

├── src/

├── results/

├── assets/

├── scripts/

└── presentation/
```

---

# Future Work

Future extensions include:

- Real-time road damage detection
- TensorRT optimization
- ONNX model export
- Jetson Nano benchmarking
- OAK-D stereo depth integration
- GPS localization
- Autonomous robot deployment
- Road condition mapping
- Infrastructure inspection platform

---

# Author

**Subir Balo**

Electronic Engineering Student

Hamm-Lippstadt University of Applied Sciences (HSHL)

Germany

---

# Acknowledgements

Special thanks to:

- NVIDIA
- Luxonis
- Ultralytics
- RT-DETR Authors
- RDD2022 Dataset Contributors
