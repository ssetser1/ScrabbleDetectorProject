from ultralytics import YOLO
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ScrabbleTrainer:
    def __init__(self, dataset_path="dataset/dataset.yaml"):
        self.dataset_path = dataset_path
        self.model = None
        self.results = None
        
    def train_model(self, epochs=100, imgsz=640, batch_size=12):
        """Train YOLOv8 model on Scrabble dataset"""
        print("Initializing YOLOv8 model...")
        
        # Load a pre-trained YOLOv8 model
        self.model = YOLO('yolov8s.pt')  
        
        print(f"Training model for {epochs} epochs...")
        print(f"Dataset: {self.dataset_path}")
        
        # Train the model
        self.results = self.model.train(
            data=self.dataset_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=20,  # Early stopping patience
            save=True,
            save_period=10,  # Save every 10 epochs
            device=0,  # Use GPU device 0
            workers=4,
            project='scrabble_detector',
            name='yolov8_scrabble'
        )
        
        print("Training completed!")
        return self.results
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        if self.model is None:
            print("No model loaded. Please train first.")
            return
        
        print("Evaluating model...")
        
        # Run validation
        metrics = self.model.val()
        
        print(f"mAP50: {metrics.box.map50:.3f}")
        print(f"mAP50-95: {metrics.box.map:.3f}")
        
        return metrics
    
    def test_on_images(self, test_images_path="dataset/test/images"):
        """Test the model on test images"""
        if self.model is None:
            print("No model loaded. Please train first.")
            return
        
        test_path = Path(test_images_path)
        if not test_path.exists():
            print(f"Test path {test_path} does not exist.")
            return
        
        print(f"Testing on images in {test_path}...")
        
        # Run inference on test images
        results = self.model.predict(
            source=str(test_path),
            save=True,
            save_txt=True,
            conf=0.25,  # Confidence threshold
            iou=0.45,   # NMS IoU threshold
            project='scrabble_detector',
            name='test_results'
        )
        
        print(f"Results saved to scrabble_detector/test_results/")
        return results
    
    def plot_training_curves(self):
        """Plot training curves"""
        if self.results is None:
            print("No training results available.")
            return
        
        print("Available keys in results_dict:", self.results.results_dict.keys())
        
        # Create plots with available metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Final metrics summary
        metrics = self.results.results_dict
        
        # mAP50
        axes[0, 0].bar(['mAP50'], [metrics['metrics/mAP50(B)']], color='blue', alpha=0.7)
        axes[0, 0].set_title('mAP50 Score')
        axes[0, 0].set_ylabel('mAP50')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        
        # mAP50-95
        axes[0, 1].bar(['mAP50-95'], [metrics['metrics/mAP50-95(B)']], color='green', alpha=0.7)
        axes[0, 1].set_title('mAP50-95 Score')
        axes[0, 1].set_ylabel('mAP50-95')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision and Recall
        axes[1, 0].bar(['Precision', 'Recall'], 
                      [metrics['metrics/precision(B)'], metrics['metrics/recall(B)']], 
                      color=['orange', 'red'], alpha=0.7)
        axes[1, 0].set_title('Precision & Recall')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Fitness score
        axes[1, 1].bar(['Fitness'], [metrics['fitness']], color='purple', alpha=0.7)
        axes[1, 1].set_title('Fitness Score')
        axes[1, 1].set_ylabel('Fitness')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('scrabble_detector/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_model(self, format='onnx'):
        """Export the trained model"""
        if self.model is None:
            print("No model loaded. Please train first.")
            return
        
        print(f"Exporting model to {format} format...")
        
        if format == 'onnx':
            self.model.export(format='onnx', dynamic=True)
        elif format == 'tflite':
            self.model.export(format='tflite')
        elif format == 'coreml':
            self.model.export(format='coreml')
        else:
            print(f"Unsupported format: {format}")
            return
        
        print(f"Model exported successfully!")

def main():
    trainer = ScrabbleTrainer()
    
    # Train the model
    print("Starting training...")
    trainer.train_model(epochs=50)  # Start with 50 epochs
    
    # Evaluate the model
    print("\nEvaluating model...")
    trainer.evaluate_model()
    
    # Test on test images
    print("\nTesting on test images...")
    trainer.test_on_images()
    
    # Plot training curves
    print("\nPlotting training curves...")
    trainer.plot_training_curves()
    
    # Export model
    print("\nExporting model...")
    trainer.export_model('onnx')
    
    print("\nTraining pipeline completed!")

if __name__ == "__main__":
    main() 