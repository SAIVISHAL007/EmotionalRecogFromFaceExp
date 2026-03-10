"""
CNN Model Evaluation Script for Facial Emotion Recognition

This script evaluates a trained CNN model on the FER-2013 test set
and generates comprehensive performance metrics.

Usage:
    python model/evaluate.py

The script will:
    1. Load the trained model
    2. Evaluate on test data
    3. Generate confusion matrix
    4. Display classification report
    5. Visualize sample predictions
    6. Calculate per-class metrics

Academic Focus:
    - Model performance analysis
    - Confusion matrix interpretation
    - Precision, recall, F1-score metrics
    - Error analysis for viva discussion
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Local imports
import config
from utils import (
    plot_confusion_matrix,
    print_classification_report,
    visualize_predictions,
    plot_training_history,
    load_training_history
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocess import load_fer2013_data


def load_trained_model(model_path):
    """
    Load the trained model from disk.
    
    Args:
        model_path (str): Path to saved model
        
    Returns:
        keras.Model: Loaded model
    """
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first: python model/train.py")
        return None
    
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!\n")
    
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels (one-hot encoded)
        
    Returns:
        tuple: (test_loss, test_accuracy, predictions)
    """
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70 + "\n")
    
    # Evaluate
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    
    print(f"\n{'='*70}")
    print(f"TEST SET RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"{'='*70}\n")
    
    # Get predictions
    print("Generating predictions...")
    predictions = model.predict(X_test, verbose=1)
    
    return test_loss, test_accuracy, predictions


def calculate_per_class_accuracy(y_true, y_pred, class_names):
    """
    Calculate accuracy for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        
    Returns:
        dict: Per-class accuracy
    """
    per_class_acc = {}
    
    print("\n" + "="*70)
    print("PER-CLASS ACCURACY")
    print("="*70)
    
    for i, class_name in enumerate(class_names):
        class_mask = (y_true == i)
        if np.sum(class_mask) > 0:
            accuracy = np.mean(y_pred[class_mask] == i)
            per_class_acc[class_name] = accuracy
            print(f"{class_name.capitalize():12s}: {accuracy*100:6.2f}%")
    
    print("="*70 + "\n")
    
    return per_class_acc


def analyze_misclassifications(X_test, y_true, y_pred, class_names, num_samples=5):
    """
    Analyze and display misclassified samples.
    
    Args:
        X_test: Test images
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        num_samples: Number of misclassified samples to show
    """
    # Find misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    print(f"\n{'='*70}")
    print(f"MISCLASSIFICATION ANALYSIS")
    print(f"{'='*70}")
    print(f"Total misclassified samples: {len(misclassified_idx)}")
    print(f"Misclassification rate: {len(misclassified_idx)/len(y_true)*100:.2f}%")
    print(f"{'='*70}\n")
    
    if len(misclassified_idx) == 0:
        print("🎉 Perfect classification! No errors found.")
        return
    
    # Show some misclassified examples
    num_samples = min(num_samples, len(misclassified_idx))
    sample_idx = np.random.choice(misclassified_idx, num_samples, replace=False)
    
    print(f"Sample misclassifications:")
    print("-" * 70)
    for idx in sample_idx:
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        print(f"Sample {idx}: True={true_label}, Predicted={pred_label}")
    print("-" * 70 + "\n")


def plot_misclassified_samples(X_test, y_true, y_pred, class_names, num_samples=9):
    """
    Visualize misclassified samples.
    
    Args:
        X_test: Test images
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        num_samples: Number of samples to display
    """
    # Find misclassified samples
    misclassified_idx = np.where(y_true != y_pred)[0]
    
    if len(misclassified_idx) == 0:
        print("No misclassified samples to display.")
        return
    
    num_samples = min(num_samples, len(misclassified_idx))
    sample_idx = np.random.choice(misclassified_idx, num_samples, replace=False)
    
    rows = int(np.sqrt(num_samples))
    cols = (num_samples + rows - 1) // rows
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_idx):
        axes[i].imshow(X_test[idx].squeeze(), cmap='gray')
        
        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]
        
        title = f"True: {true_label}\nPred: {pred_label}"
        axes[i].set_title(title, color='red', fontsize=10)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("Misclassified Samples", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    """Main evaluation pipeline."""
    
    print("\n" + "="*70)
    print("FACIAL EMOTION RECOGNITION - MODEL EVALUATION")
    print("="*70)
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Classes: {config.NUM_CLASSES}")
    print(f"Emotion labels: {config.EMOTION_LABELS}")
    print("="*70 + "\n")
    
    # ==================
    # 1. Load Dataset
    # ==================
    print("STEP 1: Loading FER-2013 Test Data")
    print("-"*70)
    
    data = load_fer2013_data()
    
    if data is None:
        print("\n❌ Failed to load dataset!")
        return
    
    X_train, y_train, X_test, y_test, class_names = data
    
    print(f"✅ Test set loaded: {len(X_test)} samples\n")
    
    # ==================
    # 2. Load Model
    # ==================
    print("\nSTEP 2: Loading Trained Model")
    print("-"*70)
    
    model = load_trained_model(config.MODEL_SAVE_PATH)
    
    if model is None:
        return
    
    model.summary()
    
    # ==================
    # 3. Evaluate Model
    # ==================
    print("\nSTEP 3: Evaluating Model")
    print("-"*70)
    
    test_loss, test_accuracy, predictions = evaluate_model(model, X_test, y_test)
    
    # Convert predictions and labels to class indices
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # ==================
    # 4. Detailed Metrics
    # ==================
    print("\nSTEP 4: Generating Detailed Metrics")
    print("-"*70)
    
    # Classification report
    print_classification_report(y_true, y_pred, class_names)
    
    # Per-class accuracy
    per_class_acc = calculate_per_class_accuracy(y_true, y_pred, class_names)
    
    # ==================
    # 5. Confusion Matrix
    # ==================
    print("\nSTEP 5: Generating Confusion Matrix")
    print("-"*70)
    
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(config.LOGS_DIR, 'confusion_matrix.png')
    )
    
    # ==================
    # 6. Error Analysis
    # ==================
    print("\nSTEP 6: Error Analysis")
    print("-"*70)
    
    analyze_misclassifications(X_test, y_true, y_pred, class_names)
    
    # ==================
    # 7. Visualizations
    # ==================
    print("\nSTEP 7: Generating Visualizations")
    print("-"*70)
    
    # Sample correct predictions
    print("Visualizing sample predictions...")
    visualize_predictions(X_test, y_true, y_pred, class_names, num_samples=16)
    
    # Misclassified samples
    print("Visualizing misclassified samples...")
    plot_misclassified_samples(X_test, y_true, y_pred, class_names, num_samples=9)
    
    # Load and plot training history if available
    history_path = config.TRAINING_HISTORY_PATH
    if os.path.exists(history_path):
        print(f"\nPlotting training history from: {history_path}")
        history = load_training_history(history_path)
        plot_training_history(history)
    
    # ==================
    # 8. Summary
    # ==================
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\n📊 Overall Performance:")
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    
    print(f"\n📈 Best Performing Classes:")
    sorted_acc = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, acc) in enumerate(sorted_acc[:3]):
        print(f"   {i+1}. {class_name.capitalize()}: {acc*100:.2f}%")
    
    print(f"\n📉 Challenging Classes:")
    for i, (class_name, acc) in enumerate(sorted_acc[-3:]):
        print(f"   {i+1}. {class_name.capitalize()}: {acc*100:.2f}%")
    
    print("\n💡 For Academic Presentation:")
    print("   - Discuss CNN architecture choices")
    print("   - Explain confusion matrix patterns")
    print("   - Address challenging emotion classes")
    print("   - Suggest improvements (data augmentation, architecture)")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
