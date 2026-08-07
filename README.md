Waste Material Classification using MobileNetV2

Project Overview

This repository contains a complete Computer Vision project forautomatic waste material classification using MobileNetV2 andtransfer learning. The system was developed as part of an academicComputer Vision course and follows a complete machine learning workflowfrom dataset preparation to model explainability.

The model classifies waste into five categories:

Cardboard

Glass

Metal

Paper

Plastic

Motivation

Automatic waste classification can improve recycling efficiency, reducehuman error and support sustainable waste management. Deep learningmodels are particularly effective because they automatically learnvisual features without requiring handcrafted descriptors.

Project Objectives

Prepare a clean and reproducible image dataset.

Train a CNN using transfer learning.

Evaluate the model with standard classification metrics.

Predict the class of unseen images.

Explain model decisions using Grad-CAM.

Build a reproducible GitHub repository.

Dataset

This project uses the Garbage Dataset -- A Comprehensive Image Datasetfor Garbage Classification and Recycling.

Dataset characteristics:

Total images: 13,348

Original classes: 10

License: MIT

Original classes:

Battery

Biological

Cardboard

Clothes

Glass

Metal

Paper

Plastic

Shoes

Trash

For this project only the following classes were selected:

Class         Images

Cardboard       1411Glass           1736Metal            930Paper           1336Plastic         1597

Dataset split:

Training: 70%

Validation: 15%

Test: 15%

Random seed: 42

Repository Structure

waste-material-classification/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
├── src/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── gradcam.py
├── README.md
├── requirements.txt
└── .gitignore

Pipeline

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

Model

Backbone: MobileNetV2

Transfer Learning:

ImageNet pretrained weights

Frozen feature extractor

Custom classification head

Classifier:

Dropout(0.3)

Linear(1280 → 5)

Training configuration:

Parameter       Value

Optimizer       AdamLoss            CrossEntropyLossLearning Rate   0.001Batch Size      32Image Size      224×224

Data Augmentation

Training images:

Resize

Random Horizontal Flip

Random Rotation

Color Jitter

ImageNet Normalization

Validation/Test:

Resize

ImageNet Normalization

Evaluation

The evaluation stage computes:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Classification Report

Final Test Results

Metric                   Score

Accuracy            82.39%Macro Precision     81.94%Macro Recall        82.68%Macro F1-score      81.97%

Generated files:

confusion_matrix.png

classification_report.csv

test_metrics.txt

test_predictions.csv

misclassified_images.csv

Prediction

The prediction module loads the trained model and classifies a singleimage, returning:

Predicted class

Confidence

Class probabilities

Prediction visualization

Grad-CAM

Grad-CAM provides visual explanations highlighting the image regionsthat most influenced the prediction.

Outputs:

Original image

Heatmap

Overlay

Combined visualization

Failure Analysis

The best-performing classes are cardboard and glass.

Most errors occur between:

Paper ↔ Cardboard

Plastic ↔ Glass

Metal ↔ Glass

These mistakes are mainly caused by similar textures, colors andlighting conditions.

Ethical Considerations

This model is intended for educational purposes.

Its performance depends on image quality and dataset diversity. Itshould not be used as the sole decision system in real industrialrecycling environments without additional validation.

Reproducibility

Create a virtual environment:

python -m venv .venv

Activate it and install dependencies:

pip install -r requirements.txt

Run the complete pipeline:

python src/prepare_data.py
python src/train.py
python src/evaluate.py
python src/predict.py
python src/gradcam.py

Future Improvements

Train on all ten classes.

Fine-tune additional MobileNetV2 layers.

Compare with EfficientNet and ResNet.

Hyperparameter optimization.

Real-time webcam inference.

Mobile deployment.

References

Garbage Dataset -- A Comprehensive Image Dataset for GarbageClassification and Recycling.

Managing Household Waste Through Transfer Learning.

PyTorch Documentation.

MobileNetV2 Paper.

Author

Computer Vision Project

Waste Material Classification using MobileNetV2

Developed for academic purposes.

Developed for academic purposes.te-material-classification
Computer Vision project for automatic waste material classification using transfer learning and MobileNetV2.
