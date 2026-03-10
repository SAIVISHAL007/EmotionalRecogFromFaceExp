"""
Configuration file for CNN model hyperparameters and training settings.
"""

import os

# ==========================
# Dataset Configuration
# ==========================
DATASET_NAME = "FER-2013"
NUM_CLASSES = 7
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Image specifications
IMAGE_HEIGHT = 48
IMAGE_WIDTH = 48
IMAGE_CHANNELS = 1  # Grayscale

# ==========================
# Model Architecture
# ==========================
# CNN Layer Configuration
CONV_LAYERS = [
    {'filters': 32, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'filters': 64, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'filters': 128, 'kernel_size': (3, 3), 'activation': 'relu'},
    {'filters': 256, 'kernel_size': (3, 3), 'activation': 'relu'},
]

POOL_SIZE = (2, 2)
DROPOUT_CONV = 0.25
DROPOUT_DENSE = 0.5

# Dense layer configuration
DENSE_UNITS = [512, 256]

# ==========================
# Training Configuration
# ==========================
BATCH_SIZE = 64
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# Optimizer
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'

# Loss function
LOSS_FUNCTION = 'categorical_crossentropy'

# Metrics
METRICS = ['accuracy']

# ==========================
# Data Augmentation
# ==========================
USE_DATA_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'rotation_range': 15,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'horizontal_flip': True,
    'zoom_range': 0.1,
    'fill_mode': 'nearest'
}

# ==========================
# Callbacks
# ==========================
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5

# ==========================
# Paths
# ==========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'trained_models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Model save path
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'emotion_cnn_model.h5')
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, 'emotion_cnn_weights.h5')
TRAINING_HISTORY_PATH = os.path.join(LOGS_DIR, 'training_history.json')

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)
