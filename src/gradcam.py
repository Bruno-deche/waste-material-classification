"""
Generate Grad-CAM visual explanations for waste material classification.

The script:
1. Loads the trained MobileNetV2 checkpoint.
2. Recreates the trained model.
3. Accepts an image path from the terminal or interactive input.
4. Predicts the waste-material class.
5. Generates a Grad-CAM heatmap for the predicted class.
6. Saves the original image, heatmap, overlay, and combined figure.
"""

# =============================================================================
# BLOCK 1 - IMPORTS, PATHS, CHECKPOINT, MODEL, AND TRANSFORM
# =============================================================================

from pathlib import Path
import argparse
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"


# =============================================================================
# CONFIGURATION
# =============================================================================

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
    Read an optional image path.

    If no image path is supplied, the user is asked to enter one
    interactively.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate a Grad-CAM explanation for a waste image "
            "classified by MobileNetV2."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=False,
        default=None,
        help=(
            "Path to the image to analyse. "
            "If omitted, the path is requested interactively."
        ),
    )

    return parser.parse_args()


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint() -> Dict:
    """Load and validate the trained model checkpoint."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}\n"
            "Run src/train.py before generating Grad-CAM results."
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
            "The number of class names does not match "
            "the num_classes value stored in the checkpoint."
        )

    return checkpoint


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_model(
    checkpoint: Dict,
) -> nn.Module:
    """Recreate MobileNetV2 and load the trained weights."""

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
# IMAGE TRANSFORM
# =============================================================================

def create_transform(
    checkpoint: Dict,
) -> transforms.Compose:
    """
    Create the same deterministic preprocessing used for evaluation.
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


# =============================================================================
# BLOCK 2 - IMAGE LOADING, PREDICTION, AND GRAD-CAM GENERATION
# =============================================================================

def request_image_path() -> Path:
    """Ask the user to enter or paste an image path."""

    print("\nNo image path was provided.")
    print("Enter the path of the image to analyse.")
    print(
        "You can also drag the image into the terminal window."
    )

    user_input = input("\nImage path: ").strip()

    if not user_input:
        raise ValueError(
            "No image path was provided."
        )

    cleaned_path = user_input.strip("\"'")

    return Path(cleaned_path)


def resolve_image_path(
    image_path: Path,
) -> Path:
    """
    Resolve an image path supplied from either src or the project root.
    """

    image_path = image_path.expanduser()

    if image_path.is_absolute():
        return image_path.resolve()

    current_directory_path = (
        Path.cwd() / image_path
    ).resolve()

    if current_directory_path.is_file():
        return current_directory_path

    project_root_path = (
        PROJECT_ROOT / image_path
    ).resolve()

    if project_root_path.is_file():
        return project_root_path

    return current_directory_path


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
            f"Unsupported image extension: {image_path.suffix}\n"
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
            f"The selected file is not a valid image: {image_path}"
        ) from error


def prepare_image_tensor(
    image: Image.Image,
    image_transform: transforms.Compose,
) -> torch.Tensor:
    """Transform the image and create a one-image batch."""

    image_tensor = image_transform(
        image
    ).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    # Grad-CAM needs gradients to propagate through the image pipeline,
    # even when the pretrained feature extractor is frozen.
    image_tensor.requires_grad_(True)

    return image_tensor


def predict_image(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
) -> Tuple[int, str, float, Dict[str, float]]:
    """Predict the class, confidence, and all class probabilities."""

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    predicted_class = class_names[
        predicted_index
    ]

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
        predicted_index,
        predicted_class,
        confidence,
        class_probabilities,
    )


def generate_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_class_index: int,
) -> Tuple[np.ndarray, str]:
    """
    Generate a Grad-CAM heatmap for the predicted class.

    The final convolutional block of MobileNetV2 is used because it
    contains the highest-level spatial feature representations.
    """

    target_layer = model.features[-1]

    target_layers = [
        target_layer
    ]

    targets = [
        ClassifierOutputTarget(
            target_class_index
        )
    ]

    with GradCAM(
        model=model,
        target_layers=target_layers,
    ) as cam:
        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=targets,
        )[0]

    target_layer_name = (
        "model.features[-1]"
    )

    return (
        grayscale_cam,
        target_layer_name,
    )


# =============================================================================
# BLOCK 3 - HEATMAP, OVERLAY, AND RESULT SAVING
# =============================================================================

def prepare_display_image(
    image: Image.Image,
    image_size: int,
) -> np.ndarray:
    """
    Resize the original image and convert it to a float RGB array.

    The returned values are in the range 0-1, as required by
    show_cam_on_image.
    """

    resized_image = image.resize(
        (image_size, image_size)
    )

    display_image = np.asarray(
        resized_image,
        dtype=np.float32,
    )

    display_image = (
        display_image / 255.0
    )

    return display_image


def create_overlay(
    display_image: np.ndarray,
    grayscale_cam: np.ndarray,
) -> np.ndarray:
    """Overlay the Grad-CAM heatmap on the original image."""

    overlay = show_cam_on_image(
        display_image,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.55,
    )

    return overlay


def save_original_image(
    display_image: np.ndarray,
    image_path: Path,
) -> Path:
    """Save the resized original image."""

    output_path = (
        RESULTS_DIR
        / f"gradcam_original_{image_path.stem}.png"
    )

    plt.imsave(
        output_path,
        display_image,
    )

    return output_path


def save_heatmap(
    grayscale_cam: np.ndarray,
    image_path: Path,
) -> Path:
    """Save the standalone Grad-CAM heatmap."""

    output_path = (
        RESULTS_DIR
        / f"gradcam_heatmap_{image_path.stem}.png"
    )

    plt.imsave(
        output_path,
        grayscale_cam,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    return output_path


def save_overlay(
    overlay: np.ndarray,
    image_path: Path,
) -> Path:
    """Save the heatmap superimposed on the original image."""

    output_path = (
        RESULTS_DIR
        / f"gradcam_overlay_{image_path.stem}.png"
    )

    plt.imsave(
        output_path,
        overlay,
    )

    return output_path


def save_combined_figure(
    display_image: np.ndarray,
    grayscale_cam: np.ndarray,
    overlay: np.ndarray,
    image_path: Path,
    predicted_class: str,
    confidence: float,
) -> Path:
    """
    Save a three-panel figure containing the original image,
    standalone heatmap, and Grad-CAM overlay.
    """

    output_path = (
        RESULTS_DIR
        / f"gradcam_combined_{image_path.stem}.png"
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].imshow(
        display_image
    )

    axes[0].set_title(
        "Original image"
    )

    axes[0].axis(
        "off"
    )

    heatmap_image = axes[1].imshow(
        grayscale_cam,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
    )

    axes[1].set_title(
        "Grad-CAM heatmap"
    )

    axes[1].axis(
        "off"
    )

    figure.colorbar(
        heatmap_image,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )

    axes[2].imshow(
        overlay
    )

    axes[2].set_title(
        f"Overlay\n"
        f"{predicted_class} "
        f"({confidence:.2%})"
    )

    axes[2].axis(
        "off"
    )

    figure.suptitle(
        "MobileNetV2 Grad-CAM Explanation",
        fontsize=16,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    return output_path


def print_probabilities(
    class_probabilities: Dict[str, float],
) -> None:
    """Print all class probabilities from highest to lowest."""

    print("\nClass probabilities:")

    sorted_probabilities = sorted(
        class_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for class_name, probability in sorted_probabilities:
        print(
            f"  {class_name:<12} "
            f"{probability:>8.2%}"
        )


# =============================================================================
# BLOCK 4 - COMPLETE MAIN PIPELINE AND FINAL OUTPUT
# =============================================================================

def main() -> None:
    """Run the complete Grad-CAM explanation pipeline."""

    print("=" * 72)
    print(
        "WASTE MATERIAL CLASSIFICATION "
        "- GRAD-CAM"
    )
    print("=" * 72)

    arguments = parse_arguments()

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

    image_tensor = prepare_image_tensor(
        image=image,
        image_transform=image_transform,
    )

    (
        predicted_index,
        predicted_class,
        confidence,
        class_probabilities,
    ) = predict_image(
        model=model,
        image_tensor=image_tensor,
        class_names=checkpoint["class_names"],
    )

    (
        grayscale_cam,
        target_layer_name,
    ) = generate_gradcam(
        model=model,
        image_tensor=image_tensor,
        target_class_index=predicted_index,
    )

    display_image = prepare_display_image(
        image=image,
        image_size=checkpoint["image_size"],
    )

    overlay = create_overlay(
        display_image=display_image,
        grayscale_cam=grayscale_cam,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_path = save_original_image(
        display_image=display_image,
        image_path=image_path,
    )

    heatmap_path = save_heatmap(
        grayscale_cam=grayscale_cam,
        image_path=image_path,
    )

    overlay_path = save_overlay(
        overlay=overlay,
        image_path=image_path,
    )

    combined_path = save_combined_figure(
        display_image=display_image,
        grayscale_cam=grayscale_cam,
        overlay=overlay,
        image_path=image_path,
        predicted_class=predicted_class,
        confidence=confidence,
    )

    print("\n" + "=" * 72)
    print("GRAD-CAM RESULT")
    print("=" * 72)

    print(f"Image: {image_path}")
    print(f"Device: {DEVICE}")
    print(
        f"Predicted class: "
        f"{predicted_class}"
    )
    print(
        f"Confidence: "
        f"{confidence:.2%}"
    )
    print(
        f"Target class index: "
        f"{predicted_index}"
    )
    print(
        f"Target layer: "
        f"{target_layer_name}"
    )
    print(
        f"Grad-CAM dimensions: "
        f"{grayscale_cam.shape}"
    )
    print(
        f"Grad-CAM value range: "
        f"{grayscale_cam.min():.4f} - "
        f"{grayscale_cam.max():.4f}"
    )

    print_probabilities(
        class_probabilities
    )

    print("\nSaved files:")
    print(f"- {original_path}")
    print(f"- {heatmap_path}")
    print(f"- {overlay_path}")
    print(f"- {combined_path}")

    print(
        "\nGrad-CAM generation completed successfully."
    )


if __name__ == "__main__":
    main()