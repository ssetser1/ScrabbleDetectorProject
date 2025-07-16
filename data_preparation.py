import os
import shutil
import yaml
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

class ScrabbleDataPreparator:
    def __init__(self, base_path="ImageData"):
        self.base_path = Path(base_path)
        self.output_path = Path("dataset")
        self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                       'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        
    def create_dataset_structure(self):
        """Create the standard YOLO dataset structure"""
        for split in ['train', 'val', 'test']:
            for subdir in ['images', 'labels']:
                (self.output_path / split / subdir).mkdir(parents=True, exist_ok=True)
        print("Dataset structure created!")
    
    def combine_datasets(self):
        """Combine real and synthetic datasets"""
        # Copy synthetic training data
        synthetic_train_images = self.base_path / "SyntheticPictures" / "images" / "train"
        synthetic_train_labels = self.base_path / "SyntheticPictures" / "labels" / "train"
        
        # Copy to main train directory
        for img_file in synthetic_train_images.glob("*.jpg"):
            shutil.copy2(img_file, self.output_path / "train" / "images")
            label_file = synthetic_train_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, self.output_path / "train" / "labels")
        
        # Copy synthetic validation data
        synthetic_val_images = self.base_path / "SyntheticPictures" / "images" / "val"
        synthetic_val_labels = self.base_path / "SyntheticPictures" / "labels" / "val"
        
        for img_file in synthetic_val_images.glob("*.jpg"):
            shutil.copy2(img_file, self.output_path / "val" / "images")
            label_file = synthetic_val_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, self.output_path / "val" / "labels")
        
        # Copy real training data
        real_train_images = self.base_path / "RealPictures" / "train" / "images"
        real_train_labels = self.base_path / "RealPictures" / "train" / "labels"
        
        for img_file in real_train_images.glob("*.jpg"):
            shutil.copy2(img_file, self.output_path / "train" / "images")
            label_file = real_train_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, self.output_path / "train" / "labels")
        
        # Copy real test data
        real_test_images = self.base_path / "RealPictures" / "test" / "images"
        real_test_labels = self.base_path / "RealPictures" / "test" / "labels"
        
        for img_file in real_test_images.glob("*.jpg"):
            shutil.copy2(img_file, self.output_path / "test" / "images")
            label_file = real_test_labels / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, self.output_path / "test" / "labels")
        
        print("Datasets combined successfully!")
    
    def create_dataset_yaml(self):
        """Create dataset.yaml file for YOLO training"""
        dataset_config = {
            'path': str(self.output_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        with open(self.output_path / "dataset.yaml", 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print("Dataset YAML created!")
    
    def validate_labels(self):
        """Validate that all label files exist and are properly formatted"""
        issues = []
        
        for split in ['train', 'val', 'test']:
            images_dir = self.output_path / split / "images"
            labels_dir = self.output_path / split / "labels"
            
            for img_file in images_dir.glob("*.jpg"):
                label_file = labels_dir / f"{img_file.stem}.txt"
                
                if not label_file.exists():
                    issues.append(f"Missing label for {img_file}")
                    continue
                
                # Validate label format
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            issues.append(f"Invalid label format in {label_file}:{line_num}")
                            continue
                        
                        class_id = int(parts[0])
                        if class_id < 0 or class_id >= len(self.classes):
                            issues.append(f"Invalid class ID {class_id} in {label_file}:{line_num}")
                        
                        # Check coordinates are normalized (0-1)
                        coords = [float(x) for x in parts[1:]]
                        if any(x < 0 or x > 1 for x in coords):
                            issues.append(f"Coordinates not normalized in {label_file}:{line_num}")
                
                except Exception as e:
                    issues.append(f"Error reading {label_file}: {e}")
        
        if issues:
            print("Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("All labels validated successfully!")
        
        return len(issues) == 0
    
    def get_dataset_stats(self):
        """Print dataset statistics"""
        stats = {}
        
        for split in ['train', 'val', 'test']:
            images_dir = self.output_path / split / "images"
            labels_dir = self.output_path / split / "labels"
            
            num_images = len(list(images_dir.glob("*.jpg")))
            num_labels = len(list(labels_dir.glob("*.txt")))
            
            # Count total annotations
            total_annotations = 0
            class_counts = {i: 0 for i in range(len(self.classes))}
            
            for label_file in labels_dir.glob("*.txt"):
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            class_counts[class_id] += 1
                            total_annotations += 1
            
            stats[split] = {
                'images': num_images,
                'labels': num_labels,
                'annotations': total_annotations,
                'class_distribution': class_counts
            }
        
        print("\nDataset Statistics:")
        print("=" * 50)
        for split, data in stats.items():
            print(f"\n{split.upper()}:")
            print(f"  Images: {data['images']}")
            print(f"  Labels: {data['labels']}")
            print(f"  Total Annotations: {data['annotations']}")
            print(f"  Avg Annotations per Image: {data['annotations']/max(data['images'], 1):.1f}")
        
        return stats

def main():
    preparator = ScrabbleDataPreparator()
    
    print("Creating dataset structure...")
    preparator.create_dataset_structure()
    
    print("Combining datasets...")
    preparator.combine_datasets()
    
    print("Creating dataset YAML...")
    preparator.create_dataset_yaml()
    
    print("Validating labels...")
    preparator.validate_labels()
    
    print("Getting dataset statistics...")
    preparator.get_dataset_stats()
    
    print("\nData preparation complete!")

if __name__ == "__main__":
    main() 