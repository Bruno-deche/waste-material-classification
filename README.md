Waste Material Classification using MobileNetV2

Project Overview

This repository contains a complete Computer Vision project for automatic waste material classification using MobileNetV2 and transfer learning.

The system was developed as part of an academic Computer Vision course and follows a complete machine learning workflow from dataset preparation to model evaluation, inference, and explainability.

The model classifies waste into five categories:

Cardboard

Glass

Metal

Paper

Plastic

Motivation

Automatic waste classification can improve recycling efficiency, reduce human error, and support sustainable waste management. Deep learning models are suitable for this task because they can automatically learn discriminative visual features directly from images without requiring handcrafted descriptors.

Project Objectives

The main objectives of the project are to:

Prepare a clean and reproducible image dataset.

Train a convolutional neural network using transfer learning.

Evaluate the model using standard classification metrics.

Predict the class of unseen images.

Explain model decisions using Grad-CAM.

Organize the implementation in a modular and reproducible GitHub repository.

Dataset

This project uses the Garbage Dataset - A Comprehensive Image Dataset for Garbage Classification and Recycling.

The complete dataset contains 13,348 images distributed across 10 classes:

Battery: 756

Biological: 699

Cardboard: 1,411

Clothes: 1,892

Glass: 1,736

Metal: 930

Paper: 1,336

Plastic: 1,597

Shoes: 1,449

Trash: 453

The dataset is distributed under the MIT License.

For this project, five material-oriented classes were selected:

Class

Images

Cardboard

1,411

Glass

1,736

Metal

930

Paper

1,336

Plastic

1,597

Total

7,010

The preprocessing script automatically creates the following split:

Split

Images

Approximate Ratio

Training

4,905

70%

Validation

1,049

15%

Test

1,056

15%

The split uses a fixed random seed of 42 for reproducibility.

The original images are expected in:

data/raw/original/

The processed dataset is generated in:

data/processed/

Repository Structure

waste-material-classification/
|
|-- data/
|   |-- raw/
|   |   `-- original/
|   `-- processed/
|       |-- train/
|       |-- validation/
|       `-- test/
|
|-- models/
|   `-- best_model.pth
|
|-- results/
|
|-- src/
|   |-- prepare_data.py
|   |-- train.py
|   |-- evaluate.py
|   |-- predict.py
|   `-- gradcam.py
|
|-- .gitignore
|-- README.md
|-- requirements.txt
`-- Technical_Analysis_Waste_Material_Classification.pdf

Computer Vision Pipeline

The application follows the required end-to-end pipeline:

Original Dataset
      |
      v
Data Acquisition and Preprocessing
      |
      v
Feature Representation
      |
      v
MobileNetV2 Classification Model
      |
      v
Post-processing and Evaluation
      |
      +--> evaluate.py
      +--> predict.py
      `--> gradcam.py

More specifically:

data/raw/original/
      |
      v
prepare_data.py
      |
      v
data/processed/
      |
      v
train.py
      |
      v
models/best_model.pth
      |
      +--> evaluate.py
      +--> predict.py
      `--> gradcam.py

Data Acquisition and Preprocessing

The prepare_data.py script:

Checks the original dataset structure.

Verifies that all selected classes are available.

Validates image files.

Creates the train, validation, and test directories.

Performs the 70/15/15 dataset split.

Uses random seed 42.

Saves dataset summaries in CSV and text format.

During the validated preprocessing run, all 7,010 selected images were valid and no corrupted images were skipped.

Data Augmentation and Input Processing

Training images use:

Resize to 224 x 224 pixels.

Random horizontal flip.

Random rotation.

Color jitter.

Tensor conversion.

ImageNet normalization.

Validation and test images use deterministic preprocessing:

Resize to 224 x 224 pixels.

Tensor conversion.

ImageNet normalization.

Random augmentation is applied only to the training set.

Feature Representation

The project uses learned CNN features rather than handcrafted descriptors.

A MobileNetV2 backbone pretrained on ImageNet is used as the convolutional feature extractor. The feature extractor is frozen during the implemented transfer-learning stage.

This approach allows the project to reuse robust visual representations learned from a large-scale image dataset while training a task-specific classifier for waste materials.

Model Architecture

Backbone

MobileNetV2

Transfer Learning Strategy

ImageNet pretrained weights.

Frozen convolutional feature extractor.

Custom five-class classification head.

The original MobileNetV2 classifier is replaced with:

Dropout(p=0.3)
Linear(1280 -> 5)

This specialized classification head satisfies the requirement that a pretrained model be combined with custom task-specific logic.

Training Configuration

Parameter

Value

Architecture

MobileNetV2

Pretraining

ImageNet

Number of classes

5

Input size

224 x 224

Batch size

32

Optimizer

Adam

Learning rate

0.001

Loss function

CrossEntropyLoss

Dropout

0.3

Random seed

42

Only parameters with requires_grad=True are passed to the optimizer.

Training and Validation

The train.py script includes:

Training and validation DataLoaders.

A complete training loop.

A separate validation loop.

Training loss and accuracy tracking.

Validation loss and accuracy tracking.

Best-model checkpoint selection.

Early-stopping logic.

Training history export.

Accuracy and loss curve generation.

The best checkpoint is selected using the lowest validation loss.

The checkpoint stores:

Model name.

Epoch.

Model state dictionary.

Optimizer state dictionary.

Validation loss.

Validation accuracy.

Class names.

Number of classes.

Image size.

ImageNet normalization values.

Random seed.

For the current saved best checkpoint:

Checkpoint Property

Value

Best checkpoint epoch

1

Validation loss

0.5838

Validation accuracy

83.41%

The checkpoint epoch indicates the epoch associated with the lowest validation loss among the executed training epochs.

Performance Evaluation

The evaluate.py script evaluates the trained model on the independent test set.

The evaluation includes all classification metrics required by the project guidelines:

Accuracy.

Precision.

Recall.

F1-score.

Confusion Matrix.

Classification Report.

Final Test Results

Metric

Score

Accuracy

82.39%

Macro Precision

81.94%

Macro Recall

82.68%

Macro F1-score

81.97%

Misclassified images

186 / 1,056

Per-Class Results

Class

Precision

Recall

F1-score

Support

Cardboard

0.8894

0.8310

0.8592

213

Glass

0.8450

0.8774

0.8609

261

Metal

0.6994

0.8643

0.7732

140

Paper

0.7877

0.8308

0.8087

201

Plastic

0.8756

0.7303

0.7964

241

Confusion Matrix

[[177   1   1  30   4]
 [  0 229  16   2  14]
 [  6   8 121   3   2]
 [ 13   6  10 167   5]
 [  3  27  25  10 176]]

The evaluation generates:

results/test_metrics.txt
results/classification_report.csv
results/confusion_matrix.png
results/test_predictions.csv
results/misclassified_images.csv

Post-processing

For classification inference, the raw model logits are converted to probabilities using softmax.

The post-processing stage:

Converts logits into class probabilities.

Selects the class with the highest probability.

Reports the prediction confidence.

Reports probabilities for all five classes.

Produces a prediction visualization.

Supports Grad-CAM-based interpretation.

This provides a clear and usable output instead of exposing raw model logits.

Single-Image Prediction

The predict.py script classifies an individual image.

It supports both command-line input and interactive path entry.

Example:

python src/predict.py --image data/processed/test/plastic/plastic_1.jpg

It reports:

Predicted class.

Confidence.

Probability of every class.

Prediction time.

Saved prediction visualization.

A validated example produced:

Predicted class: plastic
Confidence: 67.43%

Class probabilities:

plastic       67.43%
glass         26.14%
cardboard      2.27%
metal          2.14%
paper          2.02%

Grad-CAM Explainability

The gradcam.py script provides visual model explanations using Grad-CAM.

The implementation targets the final MobileNetV2 feature stage:

model.features[-1]

For each analysed image, the script generates:

Original resized image.

Grad-CAM heatmap.

Heatmap overlay.

Combined visualization.

Example output files:

results/gradcam_original_plastic_1.png
results/gradcam_heatmap_plastic_1.png
results/gradcam_overlay_plastic_1.png
results/gradcam_combined_plastic_1.png

Grad-CAM helps inspect whether the classifier is focusing on relevant regions of the waste object rather than unrelated background areas.

Failure Analysis

The model misclassified 186 of 1,056 test images.

Important confusion patterns include:

Cardboard predicted as paper: 30 cases.

Plastic predicted as glass: 27 cases.

Plastic predicted as metal: 25 cases.

Glass predicted as metal: 16 cases.

Glass predicted as plastic: 14 cases.

These errors are plausible because visually different materials can share similar appearance characteristics.

Paper and cardboard may have:

Similar flat shapes.

Printed surfaces.

Similar fibrous textures.

Plastic and glass may share:

Transparency.

Reflections.

Similar containers and shapes.

Plastic and metal may share:

Specular highlights.

Reflective packaging.

Similar object geometry.

The project saves misclassified_images.csv, allowing individual failure cases to be inspected systematically.

Ethical, Privacy, and Security Considerations

This project is intended for academic and educational purposes.

The dataset primarily contains waste objects rather than sensitive personal information, so direct privacy risk is limited. However, dataset provenance, licensing, and incidental background information should still be considered when images are collected or redistributed.

Potential limitations and biases include:

Unequal numbers of images across classes.

Dependence on lighting conditions.

Dependence on image backgrounds.

Dependence on camera characteristics.

Differences between dataset images and real deployment environments.

The current test accuracy of 82.39% is appropriate for an academic prototype but does not justify fully autonomous industrial use.

In a real recycling system, additional safeguards could include:

External validation.

Confidence thresholds.

Human review for uncertain predictions.

Continuous monitoring.

Testing on images from the actual deployment environment.

Reproducibility

1. Create a Virtual Environment

python -m venv .venv

2. Activate the Environment

On Windows PowerShell:

.venv\Scripts\Activate.ps1

3. Install Dependencies

python -m pip install -r requirements.txt

The project was successfully tested by recreating a clean virtual environment and reinstalling all dependencies from requirements.txt.

How to Run

Prepare the Dataset

python src/prepare_data.py

Train the Model

python src/train.py

Evaluate the Model

python src/evaluate.py

Predict a Single Image

Interactive mode:

python src/predict.py

Or specify an image:

python src/predict.py --image data/processed/test/plastic/plastic_1.jpg

Generate Grad-CAM

Interactive mode:

python src/gradcam.py

Or specify an image:

python src/gradcam.py --image data/processed/test/plastic/plastic_1.jpg

Generated Results

Training can generate:

results/training_history.csv
results/loss_curve.png
results/accuracy_curve.png

Dataset preparation generates:

results/dataset_summary.csv
results/dataset_summary.txt

Evaluation generates:

results/test_metrics.txt
results/classification_report.csv
results/confusion_matrix.png
results/test_predictions.csv
results/misclassified_images.csv

Prediction and Grad-CAM generate image-specific visualizations.

Technical Analysis Document

The repository includes:

Technical_Analysis_Waste_Material_Classification.pdf

The document contains:

Problem statement.

Methodology.

Experimental results.

Failure analysis.

Grad-CAM interpretability.

Limitations.

Ethical, privacy, and security considerations.

Conclusions and references.

Future Improvements

Possible future extensions include:

Train on all ten dataset classes.

Fine-tune additional MobileNetV2 layers.

Compare MobileNetV2 with EfficientNet and ResNet.

Optimize hyperparameters.

Test on an independent external dataset.

Add confidence-based rejection for uncertain predictions.

Develop real-time webcam inference.

Deploy the model as a web or mobile application.

References

Garbage Dataset - A Comprehensive Image Dataset for Garbage Classification and Recycling. MIT License.

The Garbage Dataset (GD): A Multi-Class Image Benchmark for Automated Waste Segregation.

Managing Household Waste Through Transfer Learning.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C. MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR, 2018.

Selvaraju, R. R. et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV, 2017.

PyTorch documentation.

Torchvision documentation.

Author

Computer Vision Project

Waste Material Classification using MobileNetV2

Developed for academic purposes.

