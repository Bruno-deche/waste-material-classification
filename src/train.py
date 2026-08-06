"""
Train a MobileNetV2 model for waste material classification.

Pipeline:
1. Load the prepared datasets.
2. Apply data augmentation to the training set.
3. Create PyTorch DataLoaders.
4. Load MobileNetV2 pretrained on ImageNet.
5. Freeze the convolutional backbone.
6. Replace the original classifier with a custom five-class head.
7. Train and validate the model.
8. Save the best-performing checkpoint.
9. Save training metrics and learning curves.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import csv
import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATH = MODELS_DIR / "best_model.pth"
HISTORY_CSV_PATH = RESULTS_DIR / "training_history.csv"
LOSS_CURVE_PATH = RESULTS_DIR / "loss_curve.png"
ACCURACY_CURVE_PATH = RESULTS_DIR / "accuracy_curve.png"


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 4

NUM_CLASSES = 5
RANDOM_SEED = 42

MODEL_NAME = "MobileNetV2"

NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_random_seed() -> None:
    """Set random seeds to make training as reproducible as possible."""

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# IMAGE TRANSFORMS
# =============================================================================

def create_transforms() -> Dict[str, transforms.Compose]:
    """
    Create transformations for training, validation, and testing.

    Data augmentation is applied only to the training set.
    Validation and test images receive deterministic preprocessing.
    """

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return {
        "train": train_transform,
        "validation": evaluation_transform,
        "test": evaluation_transform,
    }


# =============================================================================
# DATASET AND DATALOADER CREATION
# =============================================================================

def create_dataloaders(
    transform_dictionary: Dict[str, transforms.Compose],
) -> Tuple[
    datasets.ImageFolder,
    datasets.ImageFolder,
    datasets.ImageFolder,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """Load the datasets and create their DataLoaders."""

    required_directories = [
        DATASET_DIR / "train",
        DATASET_DIR / "validation",
        DATASET_DIR / "test",
    ]

    missing_directories = [
        directory
        for directory in required_directories
        if not directory.is_dir()
    ]

    if missing_directories:
        missing_text = "\n".join(
            str(directory) for directory in missing_directories
        )

        raise FileNotFoundError(
            "The processed dataset is incomplete.\n"
            "Run src/prepare_data.py before training.\n"
            f"Missing directories:\n{missing_text}"
        )

    train_dataset = datasets.ImageFolder(
        DATASET_DIR / "train",
        transform=transform_dictionary["train"],
    )

    validation_dataset = datasets.ImageFolder(
        DATASET_DIR / "validation",
        transform=transform_dictionary["validation"],
    )

    test_dataset = datasets.ImageFolder(
        DATASET_DIR / "test",
        transform=transform_dictionary["test"],
    )

    if train_dataset.classes != validation_dataset.classes:
        raise ValueError(
            "Training and validation class orders do not match."
        )

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(
            "Training and test class orders do not match."
        )

    if len(train_dataset.classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} classes, "
            f"but found {len(train_dataset.classes)}."
        )

    data_generator = torch.Generator()
    data_generator.manual_seed(RANDOM_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        generator=data_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader,
    )


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_model() -> nn.Module:
    """
    Create a pretrained MobileNetV2 model.

    The ImageNet feature extractor is frozen. Only the custom
    classification head is trained.
    """

    weights = models.MobileNet_V2_Weights.DEFAULT

    model = models.mobilenet_v2(weights=weights)

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    input_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(input_features, NUM_CLASSES),
    )

    return model.to(DEVICE)


# =============================================================================
# LOSS FUNCTION AND OPTIMIZER
# =============================================================================

def create_loss_function() -> nn.Module:
    """Create the cross-entropy loss for multiclass classification."""

    return nn.CrossEntropyLoss()


def create_optimizer(model: nn.Module) -> optim.Optimizer:
    """Create Adam using only trainable model parameters."""

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    return optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )


# =============================================================================
# TRAINING AND VALIDATION
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    """Train the model for one complete epoch."""

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(
            DEVICE,
            non_blocking=PIN_MEMORY,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=PIN_MEMORY,
        )

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct_predictions += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
) -> Tuple[float, float]:
    """Evaluate the model on the validation dataset."""

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(
                DEVICE,
                non_blocking=PIN_MEMORY,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=PIN_MEMORY,
            )

            outputs = model(images)
            loss = loss_function(outputs, labels)

            batch_size = images.size(0)

            running_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim=1)

            correct_predictions += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


# =============================================================================
# MODEL CHECKPOINT
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_accuracy: float,
    class_names: List[str],
) -> None:
    """Save the best model together with its required metadata."""

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_name": MODEL_NAME,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "class_names": class_names,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "random_seed": RANDOM_SEED,
    }

    torch.save(checkpoint, MODEL_PATH)


# =============================================================================
# TRAINING HISTORY
# =============================================================================

def save_training_history(
    history: Dict[str, List[float]],
) -> None:
    """Save epoch-by-epoch training metrics to a CSV file."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with HISTORY_CSV_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
            ]
        )

        for epoch_index in range(
            len(history["train_loss"])
        ):
            writer.writerow(
                [
                    epoch_index + 1,
                    history["train_loss"][epoch_index],
                    history["train_accuracy"][epoch_index],
                    history["validation_loss"][epoch_index],
                    history["validation_accuracy"][epoch_index],
                ]
            )


def save_training_curves(
    history: Dict[str, List[float]],
) -> None:
    """Save separate loss and accuracy plots."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(
        1,
        len(history["train_loss"]) + 1,
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_loss"],
        marker="o",
        label="Training loss",
    )
    plt.plot(
        epochs,
        history["validation_loss"],
        marker="o",
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        LOSS_CURVE_PATH,
        dpi=300,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        history["train_accuracy"],
        marker="o",
        label="Training accuracy",
    )
    plt.plot(
        epochs,
        history["validation_accuracy"],
        marker="o",
        label="Validation accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        ACCURACY_CURVE_PATH,
        dpi=300,
    )
    plt.close()


# =============================================================================
# COMPLETE TRAINING PROCESS
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    class_names: List[str],
) -> Dict[str, List[float]]:
    """
    Train the classifier and save the checkpoint with the lowest
    validation loss.
    """

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    epochs_without_improvement = 0

    training_start_time = time.time()

    print("\n" + "=" * 72)
    print("MODEL TRAINING")
    print("=" * 72)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
        )

        validation_loss, validation_accuracy = validate_one_epoch(
            model=model,
            dataloader=validation_loader,
            loss_function=loss_function,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        epoch_duration = time.time() - epoch_start_time

        print(
            f"\nEpoch {epoch:02d}/{NUM_EPOCHS} "
            f"- {epoch_duration:.1f} seconds"
        )

        print(
            f"Training   | "
            f"Loss: {train_loss:.4f} | "
            f"Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation | "
            f"Loss: {validation_loss:.4f} | "
            f"Accuracy: {validation_accuracy:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_accuracy = validation_accuracy
            epochs_without_improvement = 0

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                class_names=class_names,
            )

            print(
                f"Best model saved to: {MODEL_PATH}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "No validation-loss improvement. "
                f"Early stopping counter: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            print("\nEarly stopping activated.")
            break

    total_training_time = (
        time.time() - training_start_time
    )

    save_training_history(history)
    save_training_curves(history)

    print("\n" + "=" * 72)
    print("TRAINING COMPLETED")
    print("=" * 72)

    print(
        f"Total training time: "
        f"{total_training_time / 60:.2f} minutes"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.4f}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_validation_accuracy:.4f}"
    )

    print(f"Best model: {MODEL_PATH}")
    print(f"Training history: {HISTORY_CSV_PATH}")
    print(f"Loss curve: {LOSS_CURVE_PATH}")
    print(f"Accuracy curve: {ACCURACY_CURVE_PATH}")

    return history


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete model-training pipeline."""

    print("=" * 72)
    print("WASTE MATERIAL CLASSIFICATION - MODEL TRAINING")
    print("=" * 72)

    set_random_seed()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    transform_dictionary = create_transforms()

    (
        train_dataset,
        validation_dataset,
        test_dataset,
        train_loader,
        validation_loader,
        test_loader,
    ) = create_dataloaders(transform_dictionary)

    print("\nDatasets loaded successfully.")

    print(f"Training images: {len(train_dataset)}")
    print(
        f"Validation images: "
        f"{len(validation_dataset)}"
    )
    print(f"Test images: {len(test_dataset)}")

    print(
        f"Classes: "
        f"{', '.join(train_dataset.classes)}"
    )

    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Maximum epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")

    model = create_model()

    print(
        f"\n{MODEL_NAME} created successfully."
    )

    print(model.classifier)

    loss_function = create_loss_function()
    optimizer = create_optimizer(model)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if not parameter.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{trainable_parameters:,}"
    )

    print(
        f"Frozen parameters: "
        f"{frozen_parameters:,}"
    )

    train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        class_names=train_dataset.classes,
    )


if __name__ == "__main__":
    main()