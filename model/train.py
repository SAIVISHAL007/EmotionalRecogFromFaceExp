"""
CNN Model Training Script for Facial Emotion Recognition

This script trains a Convolutional Neural Network on the FER-2013 dataset
for 7-class emotion classification.

Usage:
    python model/train.py

The script will:
    1. Load and preprocess the FER-2013 dataset
    2. Build the CNN architecture
    3. Train the model with callbacks (early stopping, learning rate reduction)
    4. Save the trained model and training history
    5. Display training curves

Academic Focus:
    - CNN architecture with Conv2D, ReLU, MaxPooling, Dropout
    - Categorical cross-entropy loss
    - Adam optimizer
    - Batch normalization and regularization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

# Local imports
import config
from architecture import build_cnn_model, compile_model, get_model_summary

# Add parent directory to path for data module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocess import load_fer2013_data
from model.utils import (
    plot_training_history, save_training_history
)


def setup_callbacks(model_path, log_dir):
    """
    Setup training callbacks for model optimization.
    
    Args:
        model_path (str): Path to save the best model
        log_dir (str): Directory for TensorBoard logs
        
    Returns:
        list: List of callbacks
    """
    callbacks = []
    
    # ModelCheckpoint - Save the best model
    checkpoint = ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1,
        save_weights_only=False
    )
    callbacks.append(checkpoint)
    
    # EarlyStopping - Stop training if no improvement
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    callbacks.append(early_stop)
    
    # ReduceLROnPlateau - Reduce learning rate when plateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard - Visualization
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    callbacks.append(tensorboard)
    
    print(f"\n✅ Callbacks configured:")
    print(f"   - ModelCheckpoint: {model_path}")
    print(f"   - EarlyStopping: patience={config.EARLY_STOPPING_PATIENCE}")
    print(f"   - ReduceLROnPlateau: factor={config.REDUCE_LR_FACTOR}, patience={config.REDUCE_LR_PATIENCE}")
    print(f"   - TensorBoard: {log_dir}\n")
    
    return callbacks


def train_model(model, X_train, y_train, X_val, y_val, callbacks):
    """
    Train the CNN model.
    
    Args:
        model: Compiled Keras model
        X_train: Training images
        y_train: Training labels
        X_val: Validation images
        y_val: Validation labels
        callbacks: List of callbacks
        
    Returns:
        History: Training history object
    """
    print("\n" + "="*70)
    print("STARTING MODEL TRAINING")
    print("="*70)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print("="*70 + "\n")
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70 + "\n")
    
    return history


def split_validation_data(X_train, y_train, validation_split=0.2):
    """
    Split training data into train and validation sets.
    
    Args:
        X_train: Training images
        y_train: Training labels
        validation_split: Fraction of data for validation
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_split,
        random_state=42,
        stratify=np.argmax(y_train, axis=1)  # Stratify by class
    )
    
    print(f"\n📊 Data split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Validation: {len(X_val)} samples")
    print(f"   Validation split: {validation_split*100}%\n")
    
    return X_train, y_train, X_val, y_val


def main():
    """Main training pipeline."""
    
    print("\n" + "="*70)
    print("FACIAL EMOTION RECOGNITION - CNN TRAINING")
    print("="*70)
    print(f"Dataset: {config.DATASET_NAME}")
    print(f"Classes: {config.NUM_CLASSES}")
    print(f"Emotion labels: {config.EMOTION_LABELS}")
    print(f"Image size: {config.IMAGE_HEIGHT}x{config.IMAGE_WIDTH}")
    print("="*70 + "\n")
    
    # ==================
    # 1. Load Dataset
    # ==================
    print("STEP 1: Loading FER-2013 Dataset")
    print("-"*70)
    
    data = load_fer2013_data()
    
    if data is None:
        print("\n❌ Failed to load dataset!")
        print("Please ensure FER-2013 dataset is downloaded.")
        print("Run: python data/download_fer2013.py")
        return
    
    X_train, y_train, X_test, y_test, class_names = data
    
    # Split validation data from training data
    X_train, y_train, X_val, y_val = split_validation_data(
        X_train, y_train,
        validation_split=config.VALIDATION_SPLIT
    )
    
    # ==================
    # 2. Build Model
    # ==================
    print("\nSTEP 2: Building CNN Model")
    print("-"*70)
    
    model = build_cnn_model(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        num_classes=config.NUM_CLASSES
    )
    
    model = compile_model(model, learning_rate=config.LEARNING_RATE)
    
    get_model_summary(model)
    
    # ==================
    # 3. Setup Callbacks
    # ==================
    print("\nSTEP 3: Setting up Callbacks")
    print("-"*70)
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(config.LOGS_DIR, f"training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = setup_callbacks(
        model_path=config.MODEL_SAVE_PATH,
        log_dir=log_dir
    )
    
    # ==================
    # 4. Train Model
    # ==================
    print("\nSTEP 4: Training Model")
    print("-"*70)
    
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        callbacks=callbacks
    )
    
    # ==================
    # 5. Evaluate on Test Set
    # ==================
    print("\nSTEP 5: Evaluating on Test Set")
    print("-"*70)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    
    print(f"\n{'='*70}")
    print(f"FINAL TEST RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"{'='*70}\n")
    
    # ==================
    # 6. Save Results
    # ==================
    print("\nSTEP 6: Saving Results")
    print("-"*70)
    
    # Save training history
    history_path = os.path.join(log_dir, 'training_history.json')
    save_training_history(history, history_path)
    
    # Save final model weights
    weights_path = config.MODEL_WEIGHTS_PATH
    model.save_weights(weights_path)
    print(f"Model weights saved to: {weights_path}")
    
    # Plot training curves
    print("\nGenerating training plots...")
    plot_path = os.path.join(log_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # ==================
    # 7. Summary
    # ==================
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    print(f"✅ Model saved: {config.MODEL_SAVE_PATH}")
    print(f"✅ Weights saved: {weights_path}")
    print(f"✅ History saved: {history_path}")
    print(f"✅ Plots saved: {plot_path}")
    print(f"✅ Logs directory: {log_dir}")
    print(f"\n📊 Final Performance:")
    print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"   Test Loss: {test_loss:.4f}")
    print("\n💡 Next Steps:")
    print("   1. Run evaluation: python model/evaluate.py")
    print("   2. Test real-time detection: python realtime/webcam_app.py")
    print("   3. Start backend API: python backend/main.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n🎮 GPU Available: {len(gpus)} GPU(s) detected")
        for gpu in gpus:
            print(f"   {gpu}")
    else:
        print("\n💻 Running on CPU")
    
    # Run training
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
