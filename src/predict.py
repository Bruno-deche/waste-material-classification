"""
Predict the waste material class of a single image.

The script:
1. Loads the trained MobileNetV2 checkpoint.
2. Recreates the model architecture.
3. Accepts an image path from a command-line argument or interactive input.
4. Preprocesses the selected image.
5. Predicts the waste material class.
6. Displays the confidence score and all class probabilities.
7. Saves a visual prediction result.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import argparse
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"

PREDICTION_OUTPUT_PATH = (
    RESULTS_DIR / "prediction_result.png"
)


# =============================================================================
# PREDICTION CONFIGURATION
# =============================================================================

DEFAULT_CONFIDENCE_THRESHOLD = 0.60

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Read optional image path and confidence threshold arguments.

    If no image path is supplied, the user will be asked to enter one
    interactively.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Classify a waste image using the trained "
            "MobileNetV2 model."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=False,
        default=None,
        help=(
            "Path to the image that must be classified. "
            "If omitted, the path will be requested interactively."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help=(
            "Minimum confidence required to accept the prediction. "
            "Default: 0.60"
        ),
    )

    return parser.parse_args()


def request_image_path() -> Path:
    """Ask the user to enter or paste an image path."""

    print("\nNo image path was provided.")
    print(
        "Enter the path of the image that must be classified."
    )
    print(
        "You can also drag the image into the terminal window."
    )

    user_input = input("\nImage path: ").strip()

    if not user_input:
        raise ValueError(
            "No image path was provided."
        )

    # Remove quotes automatically added when dragging a file
    # into the Windows terminal.
    cleaned_path = user_input.strip("\"'")

    return Path(cleaned_path)


def resolve_image_path(
    image_path: Path,
) -> Path:
    """
    Resolve an image path supplied from the project root or src folder.
    """

    image_path = image_path.expanduser()

    if image_path.is_absolute():
        return image_path.resolve()

    # First try the current terminal directory.
    current_directory_path = (
        Path.cwd() / image_path
    ).resolve()

    if current_directory_path.is_file():
        return current_directory_path

    # Then try the project root.
    project_root_path = (
        PROJECT_ROOT / image_path
    ).resolve()

    if project_root_path.is_file():
        return project_root_path

    return current_directory_path


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint() -> Dict:
    """Load and validate the trained model checkpoint."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Run src/train.py before making predictions."
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

    missing_keys = required_keys.difference(
        checkpoint.keys()
    )

    if missing_keys:
        raise KeyError(
            "The checkpoint is missing required metadata: "
            + ", ".join(sorted(missing_keys))
        )

    if (
        len(checkpoint["class_names"])
        != checkpoint["num_classes"]
    ):
        raise ValueError(
            "The number of checkpoint class names does not "
            "match num_classes."
        )

    return checkpoint


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_model(
    checkpoint: Dict,
) -> nn.Module:
    """Recreate MobileNetV2 and load its trained weights."""

    num_classes = checkpoint["num_classes"]

    model = models.mobilenet_v2(
        weights=None
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(
            input_features,
            num_classes,
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def create_transform(
    checkpoint: Dict,
) -> transforms.Compose:
    """
    Create the deterministic preprocessing used during evaluation.
    """

    image_size = checkpoint["image_size"]

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=checkpoint["imagenet_mean"],
                std=checkpoint["imagenet_std"],
            ),
        ]
    )


def load_image(
    image_path: Path,
) -> Image.Image:
    """Validate, load, and convert an image to RGB."""

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if (
        image_path.suffix.lower()
        not in SUPPORTED_EXTENSIONS
    ):
        raise ValueError(
            "Unsupported image extension: "
            f"{image_path.suffix}\n"
            "Supported extensions: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )

    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as error:
        raise ValueError(
            "The selected file is not a valid image: "
            f"{image_path}"
        ) from error


# =============================================================================
# MODEL PREDICTION
# =============================================================================

def predict_image(
    model: nn.Module,
    image: Image.Image,
    image_transform: transforms.Compose,
    class_names: List[str],
) -> Tuple[
    str,
    float,
    Dict[str, float],
    float,
]:
    """
    Predict the image class and return all probabilities.

    Returns:
        predicted class
        confidence score
        probability for every class
        prediction time in seconds
    """

    image_tensor = image_transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

    prediction_time = (
        time.perf_counter() - start_time
    )

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    predicted_class = (
        class_names[predicted_index]
    )

    confidence = float(
        probabilities[predicted_index].item()
    )

    class_probabilities = {
        class_name: float(
            probabilities[index].item()
        )
        for index, class_name
        in enumerate(class_names)
    }

    return (
        predicted_class,
        confidence,
        class_probabilities,
        prediction_time,
    )


# =============================================================================
# TERMINAL OUTPUT
# =============================================================================

def print_prediction(
    image_path: Path,
    predicted_class: str,
    confidence: float,
    class_probabilities: Dict[str, float],
    confidence_threshold: float,
    prediction_time: float,
) -> None:
    """Print the prediction and all class probabilities."""

    print("\n" + "=" * 72)
    print("PREDICTION RESULT")
    print("=" * 72)

    print(f"Image: {image_path}")
    print(f"Device: {DEVICE}")
    print(
        f"Prediction time: "
        f"{prediction_time:.4f} seconds"
    )

    if confidence >= confidence_threshold:
        print(
            f"Predicted class: "
            f"{predicted_class}"
        )
        print(
            f"Confidence: "
            f"{confidence:.2%}"
        )

    else:
        print("Predicted class: uncertain")
        print(
            f"Most likely class: "
            f"{predicted_class}"
        )
        print(
            f"Confidence: "
            f"{confidence:.2%}"
        )
        print(
            "The confidence is below the accepted "
            f"threshold of "
            f"{confidence_threshold:.0%}."
        )

    print("\nClass probabilities:")

    sorted_probabilities = sorted(
        class_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for (
        class_name,
        probability,
    ) in sorted_probabilities:
        print(
            f"  {class_name:<12} "
            f"{probability:>8.2%}"
        )


# =============================================================================
# VISUAL RESULT
# =============================================================================

def save_prediction_figure(
    image: Image.Image,
    image_path: Path,
    predicted_class: str,
    confidence: float,
    confidence_threshold: float,
) -> Path:
    """
    Save the input image together with its prediction.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        RESULTS_DIR
        / f"prediction_{image_path.stem}.png"
    )

    if confidence >= confidence_threshold:
        title = (
            f"Prediction: {predicted_class}\n"
            f"Confidence: {confidence:.2%}"
        )

    else:
        title = (
            "Prediction: uncertain\n"
            f"Most likely: {predicted_class} "
            f"({confidence:.2%})"
        )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.imshow(image)
    axis.set_title(title)
    axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete single-image prediction pipeline."""

    print("=" * 72)
    print(
        "WASTE MATERIAL CLASSIFICATION "
        "- IMAGE PREDICTION"
    )
    print("=" * 72)

    arguments = parse_arguments()

    if not 0.0 <= arguments.threshold <= 1.0:
        raise ValueError(
            "The confidence threshold must be "
            "between 0 and 1."
        )

    if arguments.image is None:
        image_path = request_image_path()
    else:
        image_path = arguments.image

    image_path = resolve_image_path(
        image_path
    )

    checkpoint = load_checkpoint()

    model = create_model(
        checkpoint
    )

    image_transform = create_transform(
        checkpoint
    )

    image = load_image(
        image_path
    )

    (
        predicted_class,
        confidence,
        class_probabilities,
        prediction_time,
    ) = predict_image(
        model=model,
        image=image,
        image_transform=image_transform,
        class_names=checkpoint["class_names"],
    )

    print_prediction(
        image_path=image_path,
        predicted_class=predicted_class,
        confidence=confidence,
        class_probabilities=class_probabilities,
        confidence_threshold=arguments.threshold,
        prediction_time=prediction_time,
    )

    output_path = save_prediction_figure(
        image=image,
        image_path=image_path,
        predicted_class=predicted_class,
        confidence=confidence,
        confidence_threshold=arguments.threshold,
    )

    print(
        f"\nPrediction figure saved to: "
        f"{output_path}"
    )

    print(
        "\nPrediction completed successfully."
    )


if __name__ == "__main__":
    main()

