"""
Utility functions for data loading, preprocessing, and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import os


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy/loss curves.
    
    Args:
        history: Keras History object or dict with 'accuracy', 'val_accuracy', etc.
        save_path (str): Path to save the plot
    """
    if hasattr(history, 'history'):
        history = history.history
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_labels, save_path=None):
    """
    Plot confusion matrix for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: List of class names
        save_path (str): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def print_classification_report(y_true, y_pred, class_labels):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: List of class names
    """
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=class_labels))
    print("="*70 + "\n")


def save_training_history(history, filepath):
    """
    Save training history to JSON file.
    
    Args:
        history: Keras History object
        filepath (str): Path to save JSON file
    """
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    # Convert numpy arrays to lists for JSON serialization
    history_json = {key: [float(val) for val in values] 
                   for key, values in history_dict.items()}
    
    with open(filepath, 'w') as f:
        json.dump(history_json, f, indent=4)
    
    print(f"Training history saved to: {filepath}")


def load_training_history(filepath):
    """
    Load training history from JSON file.
    
    Args:
        filepath (str): Path to JSON file
        
    Returns:
        dict: Training history
    """
    with open(filepath, 'r') as f:
        history = json.load(f)
    
    return history


def visualize_predictions(images, true_labels, pred_labels, class_names, num_samples=16):
    """
    Visualize sample predictions with images.
    
    Args:
        images: Array of images
        true_labels: True labels
        pred_labels: Predicted labels
        class_names: List of class names
        num_samples: Number of samples to display
    """
    num_samples = min(num_samples, len(images))
    rows = int(np.sqrt(num_samples))
    cols = (num_samples + rows - 1) // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        
        true_label = class_names[true_labels[i]]
        pred_label = class_names[pred_labels[i]]
        
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        title = f"True: {true_label}\nPred: {pred_label}"
        
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def preprocess_input_image(image, target_size=(48, 48)):
    """
    Preprocess a single image for model prediction.
    
    Args:
        image: Input image (can be RGB or grayscale)
        target_size: Target size for resizing
        
    Returns:
        numpy.ndarray: Preprocessed image ready for prediction
    """
    import cv2
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize to target size
    image = cv2.resize(image, target_size)
    
    # Normalize pixel values to [0, 1]
    image = image.astype('float32') / 255.0
    
    # Reshape for model input
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    
    return image


def get_emotion_color(emotion):
    """
    Get color code for each emotion (for visualization).
    
    Args:
        emotion (str): Emotion label
        
    Returns:
        tuple: BGR color code
    """
    color_map = {
        'angry': (0, 0, 255),      # Red
        'disgust': (0, 128, 0),    # Dark Green
        'fear': (128, 0, 128),     # Purple
        'happy': (0, 255, 255),    # Yellow
        'sad': (255, 0, 0),        # Blue
        'surprise': (0, 165, 255), # Orange
        'neutral': (128, 128, 128) # Gray
    }
    return color_map.get(emotion, (255, 255, 255))


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
    print("Available functions:")
    print("  - plot_training_history()")
    print("  - plot_confusion_matrix()")
    print("  - print_classification_report()")
    print("  - save_training_history()")
    print("  - load_training_history()")
    print("  - visualize_predictions()")
    print("  - preprocess_input_image()")
    print("  - get_emotion_color()")
