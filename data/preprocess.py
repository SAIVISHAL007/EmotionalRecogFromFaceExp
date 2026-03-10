"""
Data preprocessing and loading utilities for FER-2013 dataset.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import config


def load_images_from_directory(directory, target_size=(48, 48), color_mode='grayscale'):
    """
    Load images from directory structure.
    
    Args:
        directory (str): Root directory containing class subdirectories
        target_size (tuple): Target image size
        color_mode (str): 'grayscale' or 'rgb'
        
    Returns:
        tuple: (images, labels, class_names)
    """
    images = []
    labels = []
    class_names = sorted(os.listdir(directory))
    
    print(f"Loading images from: {directory}")
    print(f"Found classes: {class_names}")
    
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        
        if not os.path.isdir(class_dir):
            continue
        
        image_files = [f for f in os.listdir(class_dir) 
                      if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"Loading {len(image_files)} images from '{class_name}'...")
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Read image
                if color_mode == 'grayscale':
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Resize
                img = cv2.resize(img, target_size)
                
                images.append(img)
                labels.append(label_idx)
                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\nTotal images loaded: {len(images)}")
    print(f"Image shape: {images.shape}")
    
    return images, labels, class_names


def preprocess_data(images, labels, num_classes=7):
    """
    Preprocess images and labels for training.
    
    Args:
        images: Array of images
        labels: Array of labels
        num_classes (int): Number of classes
        
    Returns:
        tuple: (preprocessed_images, one_hot_labels)
    """
    # Normalize pixel values to [0, 1]
    images = images.astype('float32') / 255.0
    
    # Add channel dimension if grayscale
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=-1)
    
    # Convert labels to one-hot encoding
    labels = to_categorical(labels, num_classes)
    
    print(f"Preprocessed images shape: {images.shape}")
    print(f"Preprocessed labels shape: {labels.shape}")
    
    return images, labels


def create_data_generators(augmentation=True):
    """
    Create ImageDataGenerator for training and validation.
    
    Args:
        augmentation (bool): Whether to apply data augmentation
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    if augmentation:
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=config.AUGMENTATION_CONFIG['rotation_range'],
            width_shift_range=config.AUGMENTATION_CONFIG['width_shift_range'],
            height_shift_range=config.AUGMENTATION_CONFIG['height_shift_range'],
            horizontal_flip=config.AUGMENTATION_CONFIG['horizontal_flip'],
            zoom_range=config.AUGMENTATION_CONFIG['zoom_range'],
            fill_mode=config.AUGMENTATION_CONFIG['fill_mode'],
            rescale=1./255
        )
        print("✅ Data augmentation enabled for training")
    else:
        # No augmentation
        train_datagen = ImageDataGenerator(rescale=1./255)
        print("ℹ️  Data augmentation disabled")
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen


def load_fer2013_data(data_dir=None):
    """
    Load complete FER-2013 dataset from directory structure.
    
    Args:
        data_dir (str): Path to data directory
        
    Returns:
        tuple: (X_train, y_train, X_test, y_test, class_names)
    """
    if data_dir is None:
        data_dir = config.DATA_DIR
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"❌ Dataset not found at {data_dir}")
        print("Please run 'python data/download_fer2013.py' first")
        return None
    
    print("\n" + "="*70)
    print("LOADING FER-2013 DATASET")
    print("="*70 + "\n")
    
    # Load training data
    X_train, y_train, class_names = load_images_from_directory(
        train_dir,
        target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        color_mode='grayscale'
    )
    
    print()
    
    # Load test data
    X_test, y_test, _ = load_images_from_directory(
        test_dir,
        target_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        color_mode='grayscale'
    )
    
    # Preprocess
    print("\n" + "-"*70)
    print("PREPROCESSING DATA")
    print("-"*70 + "\n")
    
    X_train, y_train = preprocess_data(X_train, y_train, config.NUM_CLASSES)
    X_test, y_test = preprocess_data(X_test, y_test, config.NUM_CLASSES)
    
    print("\n" + "="*70)
    print("DATASET LOADING COMPLETE")
    print("="*70)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Classes: {class_names}")
    print("="*70 + "\n")
    
    return X_train, y_train, X_test, y_test, class_names


if __name__ == "__main__":
    # Test data loading
    data = load_fer2013_data()
    
    if data is not None:
        X_train, y_train, X_test, y_test, class_names = data
        print("✅ Data loading successful!")
    else:
        print("❌ Failed to load data. Please check dataset availability.")
