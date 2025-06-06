# Object Detection Models Comparison in Sport With Complex Motion Patterns

This project provides a ready-to-use platform for comparing object detection models in sports environments characterized by fast motion, occlusions, low lighting, and variable visual conditions. It includes an interactive desktop app built with CustomTkinter, allowing users to test preconfigured models on custom videos or images. Initial tests have already been conducted to verify model integration and performance under challenging visual conditions.

---

## Project Objective

The goal is to evaluate and compare several state-of-the-art deep learning models for object detection, specifically in visually demanding sports footage. While the platform is complete, detection **results and benchmarks are not included yet**.

### Models supported:
- YOLOv11  
- YOLOv12  
- Faster R-CNN  
- EfficientDet  
- SSD  

### The platform enables:
- Selecting detection models via a user-friendly configuration panel  
- Loading and previewing custom video or image files  
- Choosing evaluation metrics from:  
  `precision`, `recall`, `F1`, `mAP`, `FPS`, and `detection time`  
- Toggling result visualization (on/off)  
- Customizing what to display:  
  bounding boxes, class labels, and confidence scores  
- Automatically saving results in both `.json` and `.csv` formats  
- Real-time visual feedback (if enabled) to inspect model behavior  
- Preparing for comprehensive benchmarking and comparison workflows

---

## Platform Info

Developed and tested on:

- **Python 3.12**
- PyTorch 2.6.0  
- TorchVision 0.21.0  
- Ultralytics 8.3.100  
- EfficientDet 0.4.1  
- OpenCV + contrib (4.11.0.86)  
- CustomTkinter 5.2.2 (GUI)  
- Transformers 4.50.3  
- py-cpuinfo, torchmetrics, pycocotools

> The application runs locally and supports CPU or CUDA-enabled GPU (if available).  
> This application **does not include precomputed detection metrics**. It is intended for testing and experimenting with model performance.

---

## 🚀 How to Run

1. Clone the repository and navigate into it:
```bash
git clone https://github.com/masanocki/object-detection-models-comparison.git
cd object-detection-models-comparison
```

2. Install the required packages (Python 3.12 recommended):
```bash
pip install -r requirements.txt
```

3. Launch the desktop application:
```bash
python main.py
```

> **Note:**  
> While most file and folder paths are not strictly hardcoded, the application expects a **specific directory structure** to function properly.  
> Please follow the default folder layout provided in the repository, or update the paths in the code if you change it.
