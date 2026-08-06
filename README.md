# Waste Material Classification using MobileNetV2

## Project Overview

This project implements an image classification system capable of recognizing different categories of waste materials using Deep Learning.

The model is based on **MobileNetV2** with **transfer learning** and is trained to classify waste images into five categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic

The project was developed as part of a Computer Vision course and follows a complete machine learning pipeline, including dataset preparation, model training, evaluation, prediction, and visual explanation using Grad-CAM.

---

# Objectives

The main objectives of this project are:

- Prepare a clean image dataset
- Train a convolutional neural network using transfer learning
- Evaluate the model using standard classification metrics
- Predict the class of new images
- Explain model decisions using Grad-CAM visualizations

---

# Dataset

The original dataset contains several waste categories.

For this project only the following five classes were selected:

| Class |
|-------|
| Cardboard |
| Glass |
| Metal |
| Paper |
| Plastic |

The original dataset is stored inside:

```
data/raw/original/
```

The processed dataset is automatically generated inside:

```
data/processed/
```

using the following split:

- Training: 70%
- Validation: 15%
- Test: 15%

Random seed:

```
42
```

---

# Project Structure

```
waste-material-classification/

│
├── data/
│   ├── raw/
│   │   └── original/
│   └── processed/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── models/
│   └── best_model.pth
│
├── results/
│
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── gradcam.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Project Pipeline

The complete workflow is:

```
Original Dataset
        │
        ▼
prepare_data.py
        │
        ▼
Processed Dataset
        │
        ▼
train.py
        │
        ▼
best_model.pth
        │
 ┌──────┼────────────┐
 │      │            │
 ▼      ▼            ▼
evaluate.py  predict.py  gradcam.py
```

---

# Model

Architecture:

**MobileNetV2**

Transfer learning:

- Pretrained on ImageNet
- Frozen feature extractor
- Custom classifier for 5 output classes

Classifier:

```
Dropout(0.3)
Linear(1280 → 5)
```

Loss function:

```
CrossEntropyLoss
```

Optimizer:

```
Adam
```

Learning rate:

```
0.001
```

Batch size:

```
32
```

Image size:

```
224 × 224
```

---

# Data Augmentation

Training images use the following transformations:

- Resize
- Random Horizontal Flip
- Random Rotation
- Color Jitter
- Normalization (ImageNet)

Validation and test images use only:

- Resize
- Normalization

---

# Training

The training script:

```
train.py
```

performs:

- Dataset loading
- Model creation
- Transfer learning
- Training loop
- Validation
- Best model saving
- Training history saving
- Accuracy and loss curves generation

The trained model is saved inside:

```
models/best_model.pth
```

---

# Evaluation

The evaluation script:

```
evaluate.py
```

computes:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

Generated outputs include:

- confusion_matrix.png
- classification_report.csv
- test_metrics.txt
- test_predictions.csv
- misclassified_images.csv

---

# Prediction

The prediction script:

```
predict.py
```

classifies a single image.

Example:

```
python predict.py --image path/to/image.jpg
```

If no image is specified, the script asks the user to provide one interactively.

Outputs include:

- Predicted class
- Confidence score
- Class probabilities
- Prediction figure

---

# Grad-CAM

The script:

```
gradcam.py
```

generates visual explanations for the model predictions.

Outputs include:

- Original image
- Grad-CAM heatmap
- Heatmap overlay
- Combined visualization

Grad-CAM helps interpret which image regions influenced the prediction.

---

# Results

Training generates:

- accuracy_curve.png
- loss_curve.png
- training_history.csv

Evaluation generates:

- classification_report.csv
- confusion_matrix.png
- test_metrics.txt
- test_predictions.csv
- misclassified_images.csv

Prediction generates:

- prediction_<image>.png

Grad-CAM generates:

- gradcam_original_<image>.png
- gradcam_heatmap_<image>.png
- gradcam_overlay_<image>.png
- gradcam_combined_<image>.png

---

# Requirements

Main libraries:

- Python 3.13
- PyTorch
- Torchvision
- NumPy
- Pillow
- Matplotlib
- Scikit-learn
- OpenCV
- Grad-CAM

Install all dependencies:

```
pip install -r requirements.txt
```

---

# How to Run

Prepare the dataset:

```
python src/prepare_data.py
```

Train the model:

```
python src/train.py
```

Evaluate the model:

```
python src/evaluate.py
```

Predict a new image:

```
python src/predict.py
```

Generate Grad-CAM visualization:

```
python src/gradcam.py
```

---

# Future Improvements

Possible future extensions include:

- Training on all available waste categories
- Fine-tuning additional MobileNetV2 layers
- Hyperparameter optimization
- Model comparison with EfficientNet and ResNet
- Real-time webcam classification
- Web application deployment
- Mobile deployment

---

# Author

Computer Vision Project

Waste Material Classification using MobileNetV2

Developed for academic purposes.te-material-classification
Computer Vision project for automatic waste material classification using transfer learning and MobileNetV2.
