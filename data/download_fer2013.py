"""
Script to download and prepare the FER-2013 dataset.

This script provides multiple methods to obtain the FER-2013 dataset:
1. Manual download instructions
2. Kaggle API download (if configured)
"""

import os
import sys


def print_manual_instructions():
    """Print manual download instructions."""
    print("\n" + "="*70)
    print("FER-2013 DATASET DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\nThe FER-2013 dataset needs to be downloaded manually from Kaggle.")
    print("\n📥 MANUAL DOWNLOAD STEPS:")
    print("  1. Visit: https://www.kaggle.com/datasets/msambare/fer2013")
    print("  2. Click 'Download' (requires Kaggle account)")
    print("  3. Extract the ZIP file")
    print("  4. Move the extracted folders to this 'data/' directory")
    print("\n📁 Expected directory structure after extraction:")
    print("  data/")
    print("  ├── train/")
    print("  │   ├── angry/")
    print("  │   ├── disgust/")
    print("  │   ├── fear/")
    print("  │   ├── happy/")
    print("  │   ├── sad/")
    print("  │   ├── surprise/")
    print("  │   └── neutral/")
    print("  └── test/")
    print("      └── (same structure as train)")
    print("\n" + "="*70)
    print("\n📌 ALTERNATIVE: Use Kaggle API")
    print("  1. Install: pip install kaggle")
    print("  2. Configure: Place kaggle.json in ~/.kaggle/")
    print("  3. Run: kaggle datasets download -d msambare/fer2013")
    print("  4. Extract: unzip fer2013.zip -d data/")
    print("="*70 + "\n")


def check_kaggle_api():
    """Check if Kaggle API is available and configured."""
    try:
        import kaggle
        print("✅ Kaggle API is installed and configured.")
        return True
    except (ImportError, OSError) as e:
        print("❌ Kaggle API not available or not configured.")
        print(f"   Error: {e}")
        return False


def download_with_kaggle():
    """Attempt to download dataset using Kaggle API."""
    try:
        import kaggle
        
        print("\n🔄 Downloading FER-2013 dataset from Kaggle...")
        print("This may take several minutes...\n")
        
        # Download dataset
        kaggle.api.dataset_download_files(
            'msambare/fer2013',
            path='./data',
            unzip=True
        )
        
        print("\n✅ Dataset downloaded and extracted successfully!")
        print("📁 Dataset location: ./data/")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to download dataset: {e}")
        return False


def verify_dataset_structure():
    """Verify if the dataset is properly structured."""
    required_dirs = [
        'data/train/angry',
        'data/train/disgust',
        'data/train/fear',
        'data/train/happy',
        'data/train/sad',
        'data/train/surprise',
        'data/train/neutral',
        'data/test/angry',
        'data/test/disgust',
        'data/test/fear',
        'data/test/happy',
        'data/test/sad',
        'data/test/surprise',
        'data/test/neutral'
    ]
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if not os.path.exists(full_path):
            missing_dirs.append(dir_path)
    
    if not missing_dirs:
        print("\n✅ Dataset structure verified!")
        print("All required directories are present.")
        
        # Count images in each directory
        print("\n📊 Dataset Statistics:")
        for split in ['train', 'test']:
            print(f"\n{split.upper()} SET:")
            for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
                dir_path = os.path.join(base_dir, 'data', split, emotion)
                if os.path.exists(dir_path):
                    count = len([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.png'))])
                    print(f"  {emotion.capitalize()}: {count} images")
        
        return True
    else:
        print("\n❌ Dataset structure incomplete!")
        print("Missing directories:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False


def main():
    """Main function to handle dataset download."""
    print("\n" + "="*70)
    print("FER-2013 DATASET SETUP")
    print("="*70)
    
    # First, check if dataset already exists
    if verify_dataset_structure():
        print("\n✅ Dataset is already downloaded and ready to use!")
        return
    
    print("\n📦 Dataset not found. Let's download it...\n")
    
    # Check for Kaggle API
    if check_kaggle_api():
        response = input("\nDo you want to download using Kaggle API? (y/n): ")
        if response.lower() == 'y':
            success = download_with_kaggle()
            if success:
                verify_dataset_structure()
                return
    
    # If Kaggle API fails or not available, show manual instructions
    print_manual_instructions()


if __name__ == "__main__":
    main()
