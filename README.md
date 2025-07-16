# Scrabble Tile Detector

A computer vision system that detects and recognizes Scrabble tiles from images of Scrabble boards using YOLOv8 object detection.

## Features

- **Object Detection**: Detects individual Scrabble tiles in board images
- **Letter Recognition**: Classifies each tile as one of the 26 letters (A-Z)
- **Real-time Inference**: Fast detection and classification
- **Multiple Input Sources**: Works with images, video, or webcam
- **Export Capabilities**: Export models to ONNX, TensorFlow Lite, or Core ML

## Dataset

The project uses a combination of:
- **Real Images**: 30+ manually labeled Scrabble board photos
- **Synthetic Images**: 100+ computer-generated Scrabble boards
- **Total**: ~130 training images with comprehensive annotations

## Project Structure

```
ScrabbleDetector/
├── ImageData/
│   ├── RealPictures/          # Real Scrabble board images
│   │   ├── train/
│   │   ├── test/
│   │   └── valid/
│   └── SyntheticPictures/     # Synthetic Scrabble board images
│       ├── images/
│       └── labels/
├── dataset/                   # Combined dataset (created by data_preparation.py)
├── scrabble_detector/         # Training outputs
├── data_preparation.py        # Dataset preparation script
├── train_model.py            # Model training script
├── inference.py              # Inference and detection script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ScrabbleDetector
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import ultralytics; print('Ultralytics installed successfully')"
   ```

## Quick Start

### 1. Prepare Your Dataset

First, organize and validate your dataset:

```bash
python data_preparation.py
```

This script will:
- Combine real and synthetic datasets
- Create proper YOLO format structure
- Validate all annotations
- Generate dataset statistics
- Create `dataset.yaml` configuration file

### 2. Train the Model

Train a YOLOv8 model on your Scrabble dataset:

```bash
python train_model.py
```

Training parameters can be adjusted in the script:
- `epochs`: Number of training epochs (default: 50)
- `imgsz`: Input image size (default: 640)
- `batch_size`: Batch size (default: 16)

### 3. Run Inference

Detect Scrabble tiles in new images:

```bash
python inference.py --image path/to/your/image.jpg --output result.jpg
```

Additional options:
- `--model`: Path to trained model
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold (default: 0.45)
- `--results`: Save detection results to JSON

## Usage Examples

### Basic Detection
```bash
python inference.py --image test_board.jpg
```

### Save Results
```bash
python inference.py --image test_board.jpg --output detected_board.jpg --results results.json
```

### Custom Model
```bash
python inference.py --model scrabble_detector/yolov8_scrabble/weights/best.pt --image test_board.jpg
```

## Model Architecture

The system uses **YOLOv8** (You Only Look Once version 8) with:
- **Backbone**: CSPDarknet53
- **Neck**: PANet (Path Aggregation Network)
- **Head**: Detection head with 26 classes (A-Z)
- **Input Size**: 640x640 pixels
- **Output**: Bounding boxes with letter classifications

## Training Strategy

### Data Augmentation
- Random horizontal flip
- Random rotation (±10 degrees)
- Random brightness/contrast adjustment
- Random scaling (0.5-1.5x)
- Mosaic augmentation

### Training Parameters
- **Optimizer**: AdamW
- **Learning Rate**: 0.01 with cosine annealing
- **Batch Size**: 16 (adjust based on GPU memory)
- **Epochs**: 50-100 (with early stopping)
- **Loss Function**: CIoU + BCE

### Validation
- **Metrics**: mAP50, mAP50-95, Precision, Recall
- **Validation Split**: 20% of training data
- **Early Stopping**: Patience of 20 epochs

## Performance Expectations

With the current dataset (~130 images):
- **mAP50**: 0.70-0.85 (depending on image quality)
- **Inference Speed**: 30-50 FPS on GPU
- **Accuracy**: 85-95% for clear, well-lit images

## Improving Performance

### 1. Data Quality
- Add more real-world images with varying lighting conditions
- Include images with different camera angles
- Add images with partial tile occlusion
- Include different Scrabble board designs

### 2. Data Augmentation
- Implement more aggressive augmentation for small datasets
- Use mixup and cutmix techniques
- Add synthetic noise and blur

### 3. Model Architecture
- Try larger YOLOv8 models (s, m, l, x)
- Experiment with different input resolutions
- Use ensemble methods

### 4. Post-processing
- Implement grid-based filtering
- Add Scrabble-specific constraints
- Use OCR for verification

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use smaller model (yolov8n instead of yolov8s)
   - Reduce input image size

2. **Poor Detection Accuracy**
   - Check label quality and format
   - Increase training epochs
   - Add more diverse training data
   - Adjust confidence thresholds

3. **Slow Training**
   - Use GPU acceleration
   - Reduce image resolution
   - Use smaller model variant

### Debugging

Enable verbose output during training:
```python
results = model.train(..., verbose=True)
```

Check dataset statistics:
```python
python data_preparation.py
```

## Export and Deployment

### Export Models
```python
# Export to ONNX
model.export(format='onnx')

# Export to TensorFlow Lite
model.export(format='tflite')

# Export to Core ML
model.export(format='coreml')
```

### Deployment Options
- **Web Application**: Flask/FastAPI with web interface
- **Mobile App**: TensorFlow Lite for Android/iOS
- **Desktop App**: PyQt/Tkinter GUI
- **API Service**: REST API for batch processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the detection framework
- [Roboflow](https://roboflow.com/) for annotation tools
- The open-source computer vision community

## Contact

For questions or support, please open an issue on GitHub. 