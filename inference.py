from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import argparse
import json

class ScrabbleDetector:
    def __init__(self, model_path="scrabble_detector/yolov8_scrabble/weights/best.pt"):
        self.model_path = model_path
        self.model = None
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect_tiles(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """Detect Scrabble tiles in an image"""
        if self.model is None:
            print("Model not loaded. Please load model first.")
            return None
        
        # Run inference
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        return results[0]  # Return first result
    
    def process_image(self, image_path, output_path=None, show_result=True):
        """Process an image and return detected tiles"""
        if not Path(image_path).exists():
            print(f"Image not found: {image_path}")
            return None
        
        # Detect tiles
        result = self.detect_tiles(image_path)
        if result is None:
            return None
        
        # Extract detections
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                if class_id < len(self.classes):
                    detection = {
                        'letter': self.classes[class_id],
                        'confidence': float(conf),
                        'bbox': {
                            'x1': float(box[0]),
                            'y1': float(box[1]),
                            'x2': float(box[2]),
                            'y2': float(box[3])
                        }
                    }
                    detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Visualize results
        if show_result:
            self.visualize_detections(image_path, detections, output_path)
        
        return detections
    
    def visualize_detections(self, image_path, detections, output_path=None):
        """Visualize detections on the image"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        # Draw detections
        for detection in detections:
            bbox = detection['bbox']
            letter = detection['letter']
            confidence = detection['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{letter} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Save or show result
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Result saved to: {output_path}")
        else:
            # Show image
            cv2.imshow('Scrabble Detections', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def extract_board_state(self, detections, grid_size=15):
        """Extract the current board state from detections"""
        # This is a simplified version - you might need more sophisticated grid detection
        board_state = [['' for _ in range(grid_size)] for _ in range(grid_size)]
        
        if not detections:
            return board_state
        
        # Simple grid assignment based on bounding box centers
        # This assumes the board is roughly centered and square
        for detection in detections:
            bbox = detection['bbox']
            center_x = (bbox['x1'] + bbox['x2']) / 2
            center_y = (bbox['y1'] + bbox['y2']) / 2
            
            # Convert to grid coordinates (simplified)
            # You might need to implement proper perspective transform
            grid_x = int(center_x / 640 * grid_size)  # Assuming 640x640 input
            grid_y = int(center_y / 640 * grid_size)
            
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                board_state[grid_y][grid_x] = detection['letter']
        
        return board_state
    
    def save_results(self, detections, output_path):
        """Save detection results to JSON file"""
        results = {
            'detections': detections,
            'total_tiles': len(detections),
            'letters_found': [d['letter'] for d in detections]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Scrabble Tile Detector')
    parser.add_argument('--model', type=str, default='scrabble_detector/yolov8_scrabble/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to save output image')
    parser.add_argument('--results', type=str, help='Path to save results JSON')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ScrabbleDetector(args.model)
    
    # Load model
    if not detector.load_model():
        return
    
    # Process image
    detections = detector.process_image(
        args.image, 
        output_path=args.output,
        show_result=args.output is None
    )
    
    if detections is None:
        return
    
    # Print results
    print(f"\nDetected {len(detections)} tiles:")
    for detection in detections:
        print(f"  {detection['letter']}: {detection['confidence']:.3f}")
    
    # Save results
    if args.results:
        detector.save_results(detections, args.results)
    
    # Extract board state
    board_state = detector.extract_board_state(detections)
    print("\nBoard state (simplified):")
    for row in board_state:
        print(' '.join([tile if tile else '.' for tile in row]))

if __name__ == "__main__":
    main() 