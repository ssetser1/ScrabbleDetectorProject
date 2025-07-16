from train_model import ScrabbleTrainer
from ultralytics import YOLO

def test_plotting():
    """Test the plotting function with the trained model"""
    trainer = ScrabbleTrainer()
    
    # Load the trained model
    model_path = "scrabble_detector/yolov8_scrabble3/weights/best.pt"
    trainer.model = YOLO(model_path)
    
    # Run validation to get results
    print("Running validation to get metrics...")
    metrics = trainer.model.val()
    
    # Set the results for plotting
    trainer.results = metrics
    
    # Test the plotting function
    print("Testing plotting function...")
    trainer.plot_training_curves()
    
    print("Plotting test completed!")

if __name__ == "__main__":
    test_plotting() 