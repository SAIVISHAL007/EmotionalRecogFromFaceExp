# Dataset Information

## FER-2013 Dataset

The **FER-2013 (Facial Expression Recognition 2013)** dataset is used for training the emotion recognition CNN model.

### Dataset Details

- **Total Images**: 35,887 grayscale images
- **Image Size**: 48x48 pixels
- **Color**: Grayscale (1 channel)
- **Classes**: 7 emotions
  - 0: Angry
  - 1: Disgust
  - 2: Fear
  - 3: Happy
  - 4: Sad
  - 5: Surprise
  - 6: Neutral

### Dataset Split

- **Training Set**: ~28,709 images
- **Validation Set**: ~3,589 images
- **Test Set**: ~3,589 images

### Download Instructions

The FER-2013 dataset can be obtained from:

1. **Kaggle**: https://www.kaggle.com/datasets/msambare/fer2013
2. **Original Source**: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

### Option 1: Manual Download

1. Go to https://www.kaggle.com/datasets/msambare/fer2013
2. Download the dataset (requires Kaggle account)
3. Extract the downloaded ZIP file
4. Place the extracted folders in this `data/` directory

Expected structure after extraction:
```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

### Option 2: Using Kaggle API

Install Kaggle CLI:
```bash
pip install kaggle
```

Configure API credentials (place `kaggle.json` in `~/.kaggle/`):
```bash
kaggle datasets download -d msambare/fer2013
unzip fer2013.zip -d data/
```

### Option 3: Using the download script

Run the provided download script:
```bash
python data/download_fer2013.py
```

## Data Preprocessing

The `preprocess.py` script handles:
- Loading images from directories
- Converting to grayscale (if needed)
- Resizing to 48x48
- Normalizing pixel values [0, 1]
- Creating train/validation splits
- Data augmentation (optional)

## Citation

If using FER-2013 in your work, please cite:

```
@inproceedings{goodfellow2013challenges,
  title={Challenges in representation learning: A report on three machine learning contests},
  author={Goodfellow, Ian J and Erhan, Dumitru and Carrier, Pierre Luc and Courville, Aaron and Mirza, Mehdi and Hamner, Ben and Cukierski, Will and Tang, Yichuan and Thaler, David and Lee, Dong-Hyun and others},
  booktitle={International conference on neural information processing},
  pages={117--124},
  year={2013},
  organization={Springer}
}
```

## Notes

- Ensure you have sufficient disk space (~300 MB)
- The dataset is class-imbalanced (Happy has most samples, Disgust has least)
- Consider using class weights during training
