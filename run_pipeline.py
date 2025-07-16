#!/usr/bin/env python3
"""
Scrabble Tile Detection Pipeline
Complete workflow from data preparation to model training and testing
"""

import os
import sys
from pathlib import Path
import argparse
import time
from ultralytics import YOLO

def run_data_preparation():
    """Run the data preparation step"""
    print("=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)
    
    try:
        from data_preparation import main as prep_main
        prep_main()
        print("✓ Data preparation completed successfully!")
        return True
    except Exception as e:
        print(f"✗ Data preparation failed: {e}")
        return False

def run_training(epochs=50, model_size='n'):
    """Run the model training step"""
    print("=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)
    
    try:
        from train_model import ScrabbleTrainer
        
        trainer = ScrabbleTrainer()
        
        # Start with nano model for faster training
        if model_size == 'n':
            trainer.model = YOLO('yolov8n.pt')
        elif model_size == 's':
            trainer.model = YOLO('yolov8s.pt')
        elif model_size == 'm':
            trainer.model = YOLO('yolov8m.pt')
        else:
            trainer.model = YOLO('yolov8n.pt')
        
        print(f"Training YOLOv8{model_size} model for {epochs} epochs...")
        trainer.train_model(epochs=epochs)
        
        print("Model training completed successfully!")
        return True
    except Exception as e:
        print(f"Model training failed: {e}")
        return False

def run_evaluation():
    """Run model evaluation"""
    print("=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)
    
    try:
        from train_model import ScrabbleTrainer
        
        trainer = ScrabbleTrainer()
        trainer.model = YOLO('scrabble_detector/yolov8_scrabble/weights/best.pt')
        
        # Evaluate on validation set
        metrics = trainer.evaluate_model()
        
        # Test on test images
        trainer.test_on_images()
        
        # Plot training curves
        trainer.plot_training_curves()
        
        print("✓ Model evaluation completed successfully!")
        return True
    except Exception as e:
        print(f"✗ Model evaluation failed: {e}")
        return False

def run_inference_test(test_image=None):
    """Run inference on a test image"""
    print("=" * 60)
    print("STEP 4: INFERENCE TESTING")
    print("=" * 60)
    
    try:
        from inference import ScrabbleDetector
        
        detector = ScrabbleDetector()
        
        if not detector.load_model():
            print("Failed to load model")
            return False
        
        # Use a test image if provided, otherwise use first available test image
        if test_image is None:
            test_images = list(Path("dataset/test/images").glob("*.jpg"))
            if test_images:
                test_image = str(test_images[0])
            else:
                print("No test images found")
                return False
        
        print(f"Testing on image: {test_image}")
        
        # Run detection
        detections = detector.process_image(
            test_image,
            output_path="test_result.jpg",
            show_result=False
        )
        
        if detections:
            print(f"✓ Detected {len(detections)} tiles:")
            for detection in detections[:10]:  # Show first 10
                print(f"  {detection['letter']}: {detection['confidence']:.3f}")
            if len(detections) > 10:
                print(f"  ... and {len(detections) - 10} more")
        else:
            print("No tiles detected")
        
        print("Inference testing completed!")
        return True
    except Exception as e:
        print(f"Inference testing failed: {e}")
        return False

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("=" * 60)
    print("CHECKING PREREQUISITES")
    print("=" * 60)
    
    # Check if ImageData directory exists
    if not Path("ImageData").exists():
        print("ImageData directory not found")
        return False
    
    # Check if real images exist
    real_images = list(Path("ImageData/RealPictures").rglob("*.jpg"))
    if not real_images:
        print("No real images found in ImageData/RealPictures")
        return False
    
    # Check if synthetic images exist
    synthetic_images = list(Path("ImageData/SyntheticPictures").rglob("*.jpg"))
    if not synthetic_images:
        print("No synthetic images found in ImageData/SyntheticPictures")
        return False
    
    print(f"Found {len(real_images)} real images")
    print(f"Found {len(synthetic_images)} synthetic images")
    
    # Check if required packages are installed
    try:
        import ultralytics
        print("Ultralytics installed")
    except ImportError:
        print("Ultralytics not installed. Run: pip install ultralytics")
        return False
    
    try:
        import cv2
        print("OpenCV installed")
    except ImportError:
        print("OpenCV not installed. Run: pip install opencv-python")
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Scrabble Tile Detection Pipeline')
    parser.add_argument('--skip-data-prep', action='store_true', 
                       help='Skip data preparation step')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip model training step')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip model evaluation step')
    parser.add_argument('--skip-inference', action='store_true',
                       help='Skip inference testing step')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--model-size', choices=['n', 's', 'm'], default='n',
                       help='YOLOv8 model size (n=nano, s=small, m=medium)')
    parser.add_argument('--test-image', type=str,
                       help='Path to test image for inference')
    
    args = parser.parse_args()
    
    print("SCRABBLE TILE DETECTION PIPELINE")
    print("=" * 60)
    print(f"Model Size: YOLOv8{args.model_size}")
    print(f"Training Epochs: {args.epochs}")
    print("=" * 60)
    
    start_time = time.time()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nPrerequisites not met. Please fix the issues above.")
        sys.exit(1)
    
    success_count = 0
    total_steps = 4
    
    # Step 1: Data Preparation
    if not args.skip_data_prep:
        if run_data_preparation():
            success_count += 1
    else:
        print("Skipping data preparation...")
        success_count += 1
    
    # Step 2: Model Training
    if not args.skip_training:
        if run_training(args.epochs, args.model_size):
            success_count += 1
    else:
        print("Skipping model training...")
        success_count += 1
    
    # Step 3: Model Evaluation
    if not args.skip_evaluation:
        if run_evaluation():
            success_count += 1
    else:
        print("Skipping model evaluation...")
        success_count += 1
    
    # Step 4: Inference Testing
    if not args.skip_inference:
        if run_inference_test(args.test_image):
            success_count += 1
    else:
        print("Skipping inference testing...")
        success_count += 1
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Steps completed: {success_count}/{total_steps}")
    print(f"Total time: {duration:.1f} seconds")
    
    if success_count == total_steps:
        print("All steps completed successfully!")
        print("\nNext steps:")
        print("1. Check the training results in scrabble_detector/yolov8_scrabble/")
        print("2. Test the model on your own images:")
        print("   python inference.py --image your_image.jpg")
        print("3. Export the model for deployment:")
        print("   python train_model.py (and modify to export)")
    else:
        print("Some steps failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 