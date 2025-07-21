# Scrabble Tile Detection System


> **Computer Vision System for Real-time Scrabble Tile Detection and Recognition**

An end-to-end machine learning pipeline that leverages YOLOv8 object detection to identify and classify individual Scrabble tiles from board images with high accuracy and real-time performance.

## Key Features

- **High-Accuracy Detection**: 85-95% accuracy for images
- **Multi-class Classification**: Recognizes all 26 letters (A-Z) with confidence scoring
- **Comprehensive Evaluation**: mAP50, mAP50-95, Precision, Recall metrics
- **Visualization**: Bounding box annotations with confidence scores

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP50** | 0.70-0.85 | Mean Average Precision at 50% IoU |
| **mAP50-95** | 0.65-0.80 | Mean Average Precision across IoU thresholds |
| **Precision** | 0.85-0.95 | High precision for letter classification |
| **Recall** | 0.80-0.90 | Good recall for tile detection |

## System Architecture

### Model Architecture
- **Framework**: YOLOv8 
- **Backbone**: CSPDarknet53 with PANet neck
- **Input Resolution**: 640×640 pixels
- **Output**: Bounding boxes + 26-class classification
- **Optimizer**: AdamW with cosine annealing
- **Loss Function**: CIoU + BCE

### Dataset Composition
```
Dataset Statistics:
├── Total Images: 135
├── Total Annotations: 12,217
├── Classes: 26 (A-Z)
├── Training: 113 images (10,289 annotations)
├── Validation: 20 images (1,835 annotations)
└── Testing: 1 image (93 annotations)
```


## Project Structure

```
ScrabbleDetectorProject/
├── ImageData/                    # Raw dataset organization
│   ├── RealPictures/               # 35 real-world images
│   │   ├── train/                  # Training images
│   │   ├── test/                   # Test images
│   │   └── valid/                  # Validation images
│   └── SyntheticPictures/          # 100 synthetic images
│       ├── images/                 # Generated board images
│       └── labels/                 # Corresponding annotations
├── dataset/                     # Processed YOLO dataset
├── scrabble_detector/           # Training outputs & weights
├── data_preparation.py          # Dataset pipeline
├── train_model.py              # Model training
├── inference.py                 # Real-time detection
├── run_pipeline.py              # Complete workflow
├── setup.py                     # Environment setup
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ScrabbleDetectorProject.git
cd ScrabbleDetectorProject

# Create virtual environment
python -m venv venv

venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
# Execute end-to-end pipeline
python run_pipeline.py
```

This single command will:
- Prepare and validate the dataset
- Train YOLOv8 model with early stopping
- Evaluate model performance
- Test inference on sample images
- Generate training curves and metrics

### 3. Individual Components

```bash
# Data preparation only
python data_preparation.py

# Model training only
python train_model.py

# Inference on custom image
python inference.py --image your_board.jpg --output result.jpg
```


### Custom Training Parameters

```python
from train_model import ScrabbleTrainer

trainer = ScrabbleTrainer()
results = trainer.train_model(
    epochs=100,           # Training epochs
    imgsz=640,           # Input image size
    batch_size=16,       # Batch size
    patience=20          # Early stopping patience
)
```

### Real-time Detection

```python
from inference import ScrabbleDetector

detector = ScrabbleDetector()
detections = detector.process_image(
    image_path="board.jpg",
    conf_threshold=0.25,  # Confidence threshold
    iou_threshold=0.45    # NMS IoU threshold
)
```

## Training Strategy

### Data Augmentation Pipeline
- **Geometric**: Random flip, rotation (±10°), scaling (0.5-1.5x)
- **Photometric**: Brightness, contrast, saturation adjustment
- **Advanced**: Mosaic augmentation, mixup, cutmix
- **Validation**: 20% split with stratified sampling

### Training Configuration
```yaml
Model: YOLOv8s
Optimizer: AdamW
Learning Rate: 0.01 (cosine annealing)
Batch Size: 12-16
Epochs: 50-100 (early stopping)
Loss: CIoU + BCE
Device: GPU (CUDA)
```
