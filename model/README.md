# CNN Model Training Guide

## 📚 Overview

This directory contains the core **academic component** of the facial emotion recognition system: a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.

## 🏗️ CNN Architecture

The model architecture consists of:

### Convolutional Blocks
- **Block 1**: Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
- **Block 2**: Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
- **Block 3**: Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
- **Block 4**: Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.25)

### Fully Connected Layers
- Flatten
- Dense(512) → BatchNorm → ReLU → Dropout(0.5)
- Dense(256) → BatchNorm → ReLU → Dropout(0.5)
- Dense(7, activation='softmax')

### Key Components
- **Activation**: ReLU (Rectified Linear Unit)
- **Pooling**: MaxPooling2D (2x2)
- **Regularization**: Dropout + Batch Normalization
- **Output**: Softmax (7 classes)
- **Loss**: Categorical Cross-Entropy
- **Optimizer**: Adam (learning rate: 0.001)

## 📂 Files

- `config.py` - Hyperparameters and configuration
- `architecture.py` - CNN model definition
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `utils.py` - Helper functions (visualization, preprocessing)
- `demo_architecture.py` - Quick architecture demo (no data needed)

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
# From project root
pip install -r requirements.txt
```

### Step 2: Verify Setup (Optional)

Test the architecture without data:

```bash
python model/demo_architecture.py
```

### Step 3: Download Dataset

```bash
python data/download_fer2013.py
```

Follow the instructions to download FER-2013 from Kaggle.

### Step 4: Train the Model

```bash
python model/train.py
```

Training will:
- Load and preprocess FER-2013 dataset
- Split train/validation data
- Train for up to 50 epochs (with early stopping)
- Save best model checkpoint
- Generate training curves
- Save logs and history

**Expected training time**: 30-90 minutes (depending on hardware)

### Step 5: Evaluate the Model

```bash
python model/evaluate.py
```

Evaluation provides:
- Test accuracy and loss
- Confusion matrix
- Per-class accuracy
- Classification report
- Misclassification analysis
- Sample visualizations

## 📊 Expected Performance

With the default configuration, expect:
- **Training Accuracy**: 65-75%
- **Validation Accuracy**: 60-70%
- **Test Accuracy**: 58-68%

Note: FER-2013 is challenging due to:
- Low resolution (48x48)
- Class imbalance
- Label noise
- Subjective emotion interpretation

## 🎓 Academic Focus Points

For your viva/presentation, be prepared to explain:

1. **Why CNN for this task?**
   - Spatial hierarchies in facial features
   - Translation invariance
   - Parameter sharing

2. **Architecture Choices**
   - Why 4 convolutional blocks?
   - Role of batch normalization
   - Dropout for preventing overfitting

3. **Loss Function**
   - Categorical cross-entropy for multi-class
   - Why not binary cross-entropy?

4. **Optimizer**
   - Adam vs SGD
   - Learning rate scheduling

5. **Performance Analysis**
   - Which emotions are hardest to classify?
   - Confusion between similar emotions (e.g., fear vs surprise)
   - How to improve the model?

## 🔧 Configuration

Modify `config.py` to experiment with:
- Learning rate
- Batch size
- Number of epochs
- Dropout rates
- Dense layer sizes
- Data augmentation parameters

## 📁 Output Files

After training, you'll find:

```
trained_models/
├── emotion_cnn_model.h5         # Full model (architecture + weights)
└── emotion_cnn_weights.h5       # Weights only

logs/
└── training_YYYYMMDD_HHMMSS/
    ├── training_history.json     # Training metrics
    ├── training_history.png      # Loss/accuracy curves
    └── tensorboard/              # TensorBoard logs
```

## 📈 TensorBoard

View training in real-time:

```bash
tensorboard --logdir=logs
```

Then open http://localhost:6006

## 🐛 Troubleshooting

**Out of Memory?**
- Reduce `BATCH_SIZE` in config.py
- Use a smaller model (reduce filters)

**Low Accuracy?**
- Train longer (increase EPOCHS)
- Adjust learning rate
- Enable data augmentation
- Check data quality

**Import Errors?**
- Ensure you're in the project root
- Install all dependencies
- Check Python version (3.8+)

## 💡 Tips

1. **Start small**: Test with demo_architecture.py first
2. **Monitor training**: Use TensorBoard for real-time metrics
3. **Save checkpoints**: Best model is auto-saved during training
4. **Analyze errors**: Use evaluate.py to understand misclassifications
5. **Document**: Take screenshots of results for your report

## 🎯 Next Steps

After successful training:
1. Test real-time detection: `python realtime/webcam_app.py`
2. Start backend API: `python backend/main.py`
3. Launch frontend: `cd frontend && npm run dev`
