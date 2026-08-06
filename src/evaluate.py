"""
Evaluate the trained MobileNetV2 waste classification model.

The script:
1. Loads the best saved model checkpoint.
2. Loads the test dataset.
3. Generates model predictions.
4. Computes classification metrics.
5. Saves a classification report.
6. Saves a confusion matrix.
7. Saves a CSV file containing all test predictions.
8. Saves a CSV file containing only misclassified images.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import csv
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

METRICS_TXT_PATH = RESULTS_DIR / "test_metrics.txt"
CLASSIFICATION_REPORT_CSV_PATH = (
    RESULTS_DIR / "classification_report.csv"
)
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.png"
ALL_PREDICTIONS_CSV_PATH = RESULTS_DIR / "test_predictions.csv"
MISCLASSIFIED_CSV_PATH = RESULTS_DIR / "misclassified_images.csv"


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# CHECKPOINT AND MODEL LOADING
# =============================================================================

def load_checkpoint() -> Dict:
    """Load the saved model checkpoint."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Run src/train.py before evaluation."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    required_keys = {
        "model_state_dict",
        "class_names",
        "num_classes",
        "image_size",
        "imagenet_mean",
        "imagenet_std",
    }

    missing_keys = required_keys.difference(checkpoint.keys())

    if missing_keys:
        raise KeyError(
            "The checkpoint is missing required metadata: "
            + ", ".join(sorted(missing_keys))
        )

    return checkpoint


def create_model(checkpoint: Dict) -> nn.Module:
    """Recreate MobileNetV2 and load the trained weights."""

    num_classes = checkpoint["num_classes"]

    if num_classes != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"but the checkpoint contains {num_classes}."
        )

    model = models.mobilenet_v2(weights=None)

    input_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(input_features, num_classes),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# =============================================================================
# TEST DATASET
# =============================================================================

def create_test_dataloader(
    checkpoint: Dict,
) -> Tuple[datasets.ImageFolder, DataLoader]:
    """Load the test dataset with deterministic preprocessing."""

    if not TEST_DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATASET_DIR}\n"
            "Run src/prepare_data.py before evaluation."
        )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    checkpoint["image_size"],
                    checkpoint["image_size"],
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=checkpoint["imagenet_mean"],
                std=checkpoint["imagenet_std"],
            ),
        ]
    )

    test_dataset = datasets.ImageFolder(
        TEST_DATASET_DIR,
        transform=evaluation_transform,
    )

    if test_dataset.classes != checkpoint["class_names"]:
        raise ValueError(
            "The test dataset classes do not match "
            "the classes stored in the checkpoint.\n"
            f"Dataset classes: {test_dataset.classes}\n"
            f"Checkpoint classes: {checkpoint['class_names']}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    return test_dataset, test_loader


# =============================================================================
# PREDICTION
# =============================================================================

def generate_predictions(
    model: nn.Module,
    test_loader: DataLoader,
) -> Tuple[List[int], List[int], List[float]]:
    """Generate labels, predictions, and confidence scores."""

    true_labels: List[int] = []
    predicted_labels: List[int] = []
    confidence_scores: List[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            confidences, predictions = torch.max(
                probabilities,
                dim=1,
            )

            true_labels.extend(
                labels.cpu().tolist()
            )
            predicted_labels.extend(
                predictions.cpu().tolist()
            )
            confidence_scores.extend(
                confidences.cpu().tolist()
            )

    return (
        true_labels,
        predicted_labels,
        confidence_scores,
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    true_labels: List[int],
    predicted_labels: List[int],
) -> Dict[str, float]:
    """Calculate the classification metrics required by the project."""

    return {
        "accuracy": accuracy_score(
            true_labels,
            predicted_labels,
        ),
        "precision_macro": precision_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            true_labels,
            predicted_labels,
            average="macro",
            zero_division=0,
        ),
        "precision_weighted": precision_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        ),
        "recall_weighted": recall_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        ),
        "f1_weighted": f1_score(
            true_labels,
            predicted_labels,
            average="weighted",
            zero_division=0,
        ),
    }


def save_metrics(
    metrics: Dict[str, float],
    checkpoint: Dict,
) -> None:
    """Save the main test metrics as a text file."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "WASTE MATERIAL CLASSIFICATION",
        "TEST SET EVALUATION",
        "=" * 50,
        "",
        f"Model: {checkpoint.get('model_name', 'MobileNetV2')}",
        f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}",
        f"Device: {DEVICE}",
        "",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro precision: {metrics['precision_macro']:.4f}",
        f"Macro recall: {metrics['recall_macro']:.4f}",
        f"Macro F1-score: {metrics['f1_macro']:.4f}",
        "",
        (
            "Weighted precision: "
            f"{metrics['precision_weighted']:.4f}"
        ),
        (
            "Weighted recall: "
            f"{metrics['recall_weighted']:.4f}"
        ),
        (
            "Weighted F1-score: "
            f"{metrics['f1_weighted']:.4f}"
        ),
    ]

    METRICS_TXT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# =============================================================================
# CLASSIFICATION REPORT
# =============================================================================

def save_classification_report(
    true_labels: List[int],
    predicted_labels: List[int],
    class_names: List[str],
) -> str:
    """Save per-class precision, recall, F1-score, and support."""

    report_text = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    report_dictionary = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    with CLASSIFICATION_REPORT_CSV_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "class",
                "precision",
                "recall",
                "f1_score",
                "support",
            ]
        )

        for class_name in class_names:
            class_metrics = report_dictionary[class_name]

            writer.writerow(
                [
                    class_name,
                    class_metrics["precision"],
                    class_metrics["recall"],
                    class_metrics["f1-score"],
                    class_metrics["support"],
                ]
            )

        for average_name in [
            "macro avg",
            "weighted avg",
        ]:
            average_metrics = report_dictionary[average_name]

            writer.writerow(
                [
                    average_name,
                    average_metrics["precision"],
                    average_metrics["recall"],
                    average_metrics["f1-score"],
                    average_metrics["support"],
                ]
            )

    return report_text


# =============================================================================
# CONFUSION MATRIX
# =============================================================================

def save_confusion_matrix(
    true_labels: List[int],
    predicted_labels: List[int],
    class_names: List[str],
) -> np.ndarray:
    """Create and save the confusion matrix."""

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
    )

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted class",
        ylabel="True class",
        title="Confusion Matrix - Test Set",
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = matrix.max() / 2.0

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row_index, column_index]
                    > threshold
                    else "black"
                ),
            )

    figure.tight_layout()
    figure.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return matrix


# =============================================================================
# PREDICTION REPORTS
# =============================================================================

def save_prediction_reports(
    test_dataset: datasets.ImageFolder,
    true_labels: List[int],
    predicted_labels: List[int],
    confidence_scores: List[float],
    class_names: List[str],
) -> int:
    """Save all predictions and the misclassified-image subset."""

    if not (
        len(test_dataset.samples)
        == len(true_labels)
        == len(predicted_labels)
        == len(confidence_scores)
    ):
        raise ValueError(
            "Prediction results do not match "
            "the number of test images."
        )

    all_rows = []
    incorrect_rows = []

    for index, (image_path, _) in enumerate(
        test_dataset.samples
    ):
        true_index = true_labels[index]
        predicted_index = predicted_labels[index]

        row = {
            "image_path": image_path,
            "true_class": class_names[true_index],
            "predicted_class": class_names[predicted_index],
            "confidence": confidence_scores[index],
            "correct": true_index == predicted_index,
        }

        all_rows.append(row)

        if true_index != predicted_index:
            incorrect_rows.append(row)

    fieldnames = [
        "image_path",
        "true_class",
        "predicted_class",
        "confidence",
        "correct",
    ]

    with ALL_PREDICTIONS_CSV_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_rows)

    with MISCLASSIFIED_CSV_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(incorrect_rows)

    return len(incorrect_rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete test-set evaluation pipeline."""

    print("=" * 72)
    print("WASTE MATERIAL CLASSIFICATION - MODEL EVALUATION")
    print("=" * 72)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint()
    model = create_model(checkpoint)

    test_dataset, test_loader = create_test_dataloader(
        checkpoint
    )

    print(f"\nModel loaded from: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    print(f"Test images: {len(test_dataset)}")
    print(
        f"Classes: "
        f"{', '.join(checkpoint['class_names'])}"
    )

    print("\nGenerating test predictions...")

    (
        true_labels,
        predicted_labels,
        confidence_scores,
    ) = generate_predictions(
        model,
        test_loader,
    )

    metrics = calculate_metrics(
        true_labels,
        predicted_labels,
    )

    save_metrics(
        metrics,
        checkpoint,
    )

    report_text = save_classification_report(
        true_labels,
        predicted_labels,
        checkpoint["class_names"],
    )

    matrix = save_confusion_matrix(
        true_labels,
        predicted_labels,
        checkpoint["class_names"],
    )

    incorrect_count = save_prediction_reports(
        test_dataset,
        true_labels,
        predicted_labels,
        confidence_scores,
        checkpoint["class_names"],
    )

    print("\n" + "=" * 72)
    print("TEST RESULTS")
    print("=" * 72)

    print(
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Macro precision: "
        f"{metrics['precision_macro']:.4f}"
    )

    print(
        f"Macro recall: "
        f"{metrics['recall_macro']:.4f}"
    )

    print(
        f"Macro F1-score: "
        f"{metrics['f1_macro']:.4f}"
    )

    print(
        f"Misclassified images: "
        f"{incorrect_count}/{len(test_dataset)}"
    )

    print("\nClassification report:\n")
    print(report_text)

    print("Confusion matrix:\n")
    print(matrix)

    print("\nSaved files:")
    print(f"- {METRICS_TXT_PATH}")
    print(f"- {CLASSIFICATION_REPORT_CSV_PATH}")
    print(f"- {CONFUSION_MATRIX_PATH}")
    print(f"- {ALL_PREDICTIONS_CSV_PATH}")
    print(f"- {MISCLASSIFIED_CSV_PATH}")

    print("\nModel evaluation completed successfully.")


if __name__ == "__main__":
    main()