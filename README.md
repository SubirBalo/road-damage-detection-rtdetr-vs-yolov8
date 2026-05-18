# road-damage-detection-rtdetr-vs-yolov8
Comparative road damage detection project using RT-DETR-L and YOLOv8-L on the RDD2022 dataset
- **RT-DETR-L**
- **YOLOv8-L**

The goal is to evaluate and compare both models using the same dataset, training setup, and evaluation metrics, and then present the results in a clear and reproducible way.

---

## Project Overview

This repository contains:

- dataset organization notes
- training notebooks and experiments
- model comparison results
- plots and visualizations
- documentation for reproducibility

The project focuses on:

- training both detectors on road damage data
- comparing overall and per-class detection performance
- analyzing metrics such as Precision, Recall, mAP50, and mAP50-95
- visualizing results with tables, charts, and training curves

---

## Models Used

### RT-DETR-L
RT-DETR-L is a transformer-based real-time object detector designed to provide strong detection accuracy with end-to-end object prediction.

### YOLOv8-L
YOLOv8-L is a one-stage object detector known for fast and strong detection performance in practical computer vision tasks.

---

## Dataset

This project uses the **RDD 2022 dataset** for road damage detection.

Classes used in this work:

- D00
- D10
- D20
- D40
- Other

---

## Evaluation Metrics

The following metrics are used for comparison:

- Precision
- Recall
- mAP50
- mAP50-95

---

## Repository Structure

```text
road-damage-detection-rtdetr-yolo/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── data/
├── notebooks/
├── src/
├── results/
├── models/
├── docs/
└── assets/




Current Best Results
YOLOv8-L
Precision: 0.670
Recall: 0.603
mAP50: 0.643
mAP50-95: 0.357



RT-DETR-L
Precision: 0.696
Recall: 0.602
mAP50: 0.643
mAP50-95: 0.331



Main Observation
YOLOv8-L achieved a higher mAP50-95
RT-DETR-L achieved a slightly higher Precision
Both models achieved nearly identical mAP50
YOLOv8-L performed better overall for this project based on localization-quality-sensitive evaluation 


Author

Subir Balo
Electronic Engineering Student