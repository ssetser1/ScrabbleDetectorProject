from train_model import ScrabbleTrainer
from ultralytics import YOLO

if __name__ == "__main__":
    # Path to the best trained model (update if needed)
    model_path = "scrabble_detector/yolov8_scrabble3/weights/best.pt"
    
    # Initialize the trainer and load the model
    trainer = ScrabbleTrainer()
    trainer.model = YOLO(model_path)
    
    # Run inference on all test images
    print("Running inference on test set...")
    results = trainer.test_on_images(test_images_path="dataset/test/images")
    print("Test set inference complete!")
    print("Results saved to scrabble_detector/test_results/") 