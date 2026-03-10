"""
Quick demo to test CNN architecture without training.

This script demonstrates the model architecture and can be used
to verify the setup before full training.

Usage:
    python model/demo_architecture.py
"""

import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import config
from model.architecture import build_cnn_model, compile_model, get_model_summary


def demo_architecture():
    """Demonstrate the CNN architecture."""
    
    print("\n" + "="*70)
    print("CNN ARCHITECTURE DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows the model architecture without requiring data.\n")
    
    # Build model
    print("Building CNN model...")
    model = build_cnn_model(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        num_classes=config.NUM_CLASSES
    )
    
    # Compile model
    model = compile_model(model, learning_rate=config.LEARNING_RATE)
    
    # Display summary
    get_model_summary(model)
    
    # Show configuration
    print("\n" + "="*70)
    print("MODEL CONFIGURATION")
    print("="*70)
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Input shape: ({config.IMAGE_HEIGHT}, {config.IMAGE_WIDTH}, {config.IMAGE_CHANNELS})")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Emotion labels: {config.EMOTION_LABELS}")
    print(f"\nTraining Configuration:")
    print(f"  - Optimizer: {config.OPTIMIZER}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Loss function: {config.LOSS_FUNCTION}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Epochs: {config.EPOCHS}")
    print(f"  - Validation split: {config.VALIDATION_SPLIT}")
    print("="*70 + "\n")
    
    # Test model with dummy data
    print("Testing model with dummy input...")
    dummy_input = np.random.rand(1, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS)
    dummy_output = model.predict(dummy_input, verbose=0)
    
    print(f"✅ Model test successful!")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {dummy_output.shape}")
    print(f"   Output probabilities (7 classes): {dummy_output[0]}")
    print(f"   Sum of probabilities: {np.sum(dummy_output[0]):.4f} (should be 1.0)")
    print(f"   Predicted class: {np.argmax(dummy_output[0])} ({config.EMOTION_LABELS[np.argmax(dummy_output[0])]})")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Download FER-2013 dataset:")
    print("   → python data/download_fer2013.py")
    print("\n2. Train the model:")
    print("   → python model/train.py")
    print("\n3. Evaluate the model:")
    print("   → python model/evaluate.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        demo_architecture()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
