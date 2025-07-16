#!/usr/bin/env python3
"""
Setup script for Scrabble Tile Detector
Installs dependencies and verifies the environment
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements = [
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "albumentations>=1.3.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0"
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            return False
    
    return True

def verify_installation():
    """Verify that all packages are installed correctly"""
    print("\nVerifying installation...")
    
    packages = [
        ("ultralytics", "Ultralytics YOLOv8"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("pandas", "Pandas"),
        ("albumentations", "Albumentations"),
        ("tqdm", "TQDM"),
        ("yaml", "PyYAML")
    ]
    
    all_good = True
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name} imported successfully")
        except ImportError:
            print(f"✗ {name} import failed")
            all_good = False
    
    return all_good

def check_dataset():
    """Check if the dataset is properly organized"""
    print("\nChecking dataset structure...")
    
    required_paths = [
        "ImageData/RealPictures/train/images",
        "ImageData/RealPictures/train/labels",
        "ImageData/RealPictures/test/images",
        "ImageData/RealPictures/test/labels",
        "ImageData/SyntheticPictures/images/train",
        "ImageData/SyntheticPictures/images/val",
        "ImageData/SyntheticPictures/labels/train",
        "ImageData/SyntheticPictures/labels/val"
    ]
    
    all_good = True
    
    for path in required_paths:
        if Path(path).exists():
            # Count files
            if "images" in path:
                files = list(Path(path).glob("*.jpg"))
            else:
                files = list(Path(path).glob("*.txt"))
            
            print(f"✓ {path}: {len(files)} files")
        else:
            print(f"✗ {path}: Not found")
            all_good = False
    
    return all_good

def main():
    print("SCRABBLE TILE DETECTOR SETUP")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("\n✗ Failed to install some packages. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Some packages failed to import. Please try reinstalling.")
        sys.exit(1)
    
    # Check dataset
    if not check_dataset():
        print("\n✗ Dataset structure is incomplete. Please ensure all required directories and files exist.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("✓ SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\nYou can now run the pipeline:")
    print("  python run_pipeline.py")
    print("\nOr run individual steps:")
    print("  python data_preparation.py")
    print("  python train_model.py")
    print("  python inference.py --image your_image.jpg")

if __name__ == "__main__":
    main() 