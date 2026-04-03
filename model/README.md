# 🧠 Model Training

CNN model training for the Facial Emotion Recognition System.

---

## Files

| File | Purpose |
|------|---------|
| `train.py` | Local CPU training — FER-2013 dataset |
| `train_rf_colab.ipynb` | Colab notebook for the optional MediaPipe + Random Forest workflow |
| `architecture.py` | 4-block CNN definition |
| `config.py` | Hyperparameters, paths, emotion labels |
| `evaluate.py` | Confusion matrix, classification report |
| `utils.py` | Data loading, augmentation, plotting |

---

## Option 1: Local CNN Training

```bash
python model/train.py
```

**Output**: `trained_models/emotion_cnn_model.h5`

### Expected performance (FER-2013)

| Metric | Value |
|--------|-------|
| Training accuracy | ~65-70% |
| Validation accuracy | **~55%** (local CPU ceiling) |
| Training time (CPU) | 30–90 min |
| Training time (GPU) | 10–20 min |

> FER-2013 is intentionally challenging — human accuracy is only ~65±5%. The CNN's 55% is the standard baseline for this dataset.

### Architecture

```
Input: 48×48×1 grayscale
│
Conv Block 1: Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
Conv Block 2: Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
Conv Block 3: Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
Conv Block 4: Conv2D(256, 3×3) → BatchNorm → ReLU → MaxPool(2×2) → Dropout(0.25)
│
Flatten
Dense(512) → BatchNorm → ReLU → Dropout(0.5)
Dense(256) → BatchNorm → ReLU → Dropout(0.5)
Dense(7)   → Softmax
```

### Hyperparameters (`config.py`)

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 50 |
| Optimizer | Adam |
| Loss | Categorical Cross-Entropy |
| Early stopping patience | 10 epochs |
| LR decay patience | 5 epochs |

---

## Option 2: Google Colab — Random Forest (Optional Alternate Workflow)

Open `model/train_rf_colab.ipynb` in Google Colab:

1. Upload the notebook to Colab
2. Select **Runtime → GPU** (T4 is free)
3. Upload `fer2013.zip` to Colab or mount Google Drive
4. Run all cells
5. Download `emotion_rf_model.pkl`
6. Place in `trained_models/`

**Why the alternate RF path exists:**

| Aspect | CNN (local) | RF + MediaPipe (Colab) |
|--------|------------|------------------------|
| Accuracy | ~55% | Varies by run and feature selection |
| Input | Raw 48×48 pixels | 478 3D landmarks |
| Imbalance handling | SMOTE optional | SMOTE applied |
| Inference speed | ~80ms/face | ~5ms/face |
| Python environment | Tested with the current repo runtime | Optional path may vary by dependency version |

---

## Dataset Setup (FER-2013)

```bash
# Option A: Kaggle CLI
kaggle datasets download msambare/fer2013 -p data/

# Option B: Manual
# Download fer2013.zip from https://www.kaggle.com/datasets/msambare/fer2013
# Place in project root, then:
python extract_dataset.py
```

Expected structure after extraction:
```
data/
  fer2013/
    train/
      angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
    validation/
      angry/ disgust/ fear/ happy/ neutral/ sad/ surprise/
```

---

## Evaluate Trained Model

```bash
python model/evaluate.py
```

Outputs:
- Confusion matrix heatmap
- Per-class precision, recall, F1
- Sample predictions grid
- Overall accuracy
