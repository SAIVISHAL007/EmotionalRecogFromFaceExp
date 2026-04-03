"""
CNN Architecture for Facial Emotion Recognition

This module defines the Convolutional Neural Network architecture
designed for the FER-2013 dataset (7-class emotion classification).

Architecture:
    - Multiple Conv2D layers with ReLU activation
    - MaxPooling2D for spatial dimension reduction
    - Dropout for regularization
    - Fully connected Dense layers
    - Softmax output for multi-class classification
"""

from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, Activation
)
import config


def build_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Build and compile the CNN model for emotion recognition.
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of emotion classes to predict
        
    Returns:
        keras.Model: Compiled CNN model
    """
    
    model = models.Sequential(name='EmotionCNN')
    model.add(keras.Input(shape=input_shape, name='input'))
    
    # ==================
    # Block 1: Conv + Pool
    # ==================
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', name='conv1'))
    model.add(BatchNormalization(name='bn1'))
    model.add(Activation('relu', name='relu1'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
    model.add(Dropout(0.25, name='dropout1'))
    
    # ==================
    # Block 2: Conv + Pool
    # ==================
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', name='conv2'))
    model.add(BatchNormalization(name='bn2'))
    model.add(Activation('relu', name='relu2'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
    model.add(Dropout(0.25, name='dropout2'))
    
    # ==================
    # Block 3: Conv + Pool
    # ==================
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', name='conv3'))
    model.add(BatchNormalization(name='bn3'))
    model.add(Activation('relu', name='relu3'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
    model.add(Dropout(0.25, name='dropout4'))
    
    # ==================
    # Block 4: Conv + Pool
    # ==================
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', name='conv4'))
    model.add(BatchNormalization(name='bn4'))
    model.add(Activation('relu', name='relu4'))
    model.add(MaxPooling2D(pool_size=(2, 2), name='pool4'))
    model.add(Dropout(0.25, name='dropout4b'))
    
    # ==================
    # Flatten & Dense Layers
    # ==================
    model.add(Flatten(name='flatten'))
    
    model.add(Dense(512, name='fc1'))
    model.add(BatchNormalization(name='bn5'))
    model.add(Activation('relu', name='relu5'))
    model.add(Dropout(0.5, name='dropout5'))
    
    model.add(Dense(256, name='fc2'))
    model.add(BatchNormalization(name='bn6'))
    model.add(Activation('relu', name='relu6'))
    model.add(Dropout(0.5, name='dropout6'))
    
    # ==================
    # Output Layer (Softmax)
    # ==================
    model.add(Dense(num_classes, activation='softmax', name='output'))
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer, loss, and metrics.
    
    Args:
        model (keras.Model): The model to compile
        learning_rate (float): Learning rate for the optimizer
        
    Returns:
        keras.Model: Compiled model
    """
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def get_model_summary(model):
    """
    Print comprehensive model summary.
    
    Args:
        model (keras.Model): The model to summarize
    """
    print("\n" + "="*70)
    print("CNN MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    model.summary()
    print("="*70)
    print(f"Total Parameters: {model.count_params():,}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test the architecture
    print("Building CNN model...")
    
    model = build_cnn_model(
        input_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNELS),
        num_classes=config.NUM_CLASSES
    )
    
    model = compile_model(model, learning_rate=config.LEARNING_RATE)
    
    get_model_summary(model)
    
    print("Model built successfully!")
    print(f"Input shape: ({config.IMAGE_HEIGHT}, {config.IMAGE_WIDTH}, {config.IMAGE_CHANNELS})")
    print(f"Output classes: {config.NUM_CLASSES}")
    print(f"Emotion labels: {config.EMOTION_LABELS}")
