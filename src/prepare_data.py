"""
Prepare the waste image dataset for model training.

The script:
1. Checks the original dataset structure.
2. Selects the five material classes used in the project.
3. Verifies that image files can be opened.
4. Splits each class into training, validation, and test sets.
5. Copies the images into data/processed.
6. Prints and saves a final dataset summary.

Dataset split:
- Training: 70%
- Validation: 15%
- Test: 15%
"""

from pathlib import Path
import csv
import random
import shutil
from typing import Dict, List, Tuple

from PIL import Image, UnidentifiedImageError


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "data" / "raw" / "original"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

SUMMARY_CSV_PATH = RESULTS_DIR / "dataset_summary.csv"
SUMMARY_TXT_PATH = RESULTS_DIR / "dataset_summary.txt"

SELECTED_CLASSES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
]

SPLIT_NAMES = ("train", "validation", "test")

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def check_configuration() -> None:
    """Validate project paths, class folders, and split ratios."""

    print("\nChecking dataset structure...")

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {SOURCE_DIR}\n"
            "Copy the original dataset into data/raw/original."
        )

    if not SOURCE_DIR.is_dir():
        raise NotADirectoryError(
            f"The dataset path is not a directory: {SOURCE_DIR}"
        )

    print(f"[OK] Dataset directory found: {SOURCE_DIR}")

    print("\nChecking selected classes...")

    missing_classes = []

    for class_name in SELECTED_CLASSES:
        class_directory = SOURCE_DIR / class_name

        if class_directory.is_dir():
            print(f"[OK] {class_name}")
        else:
            print(f"[MISSING] {class_name}")
            missing_classes.append(class_name)

    if missing_classes:
        raise FileNotFoundError(
            "The following required class directories are missing: "
            + ", ".join(missing_classes)
        )

    ratio_sum = TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO

    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, but their sum is {ratio_sum}."
        )

    print("\n[OK] Split ratios are valid.")


def reset_output_directory() -> None:
    """
    Recreate the processed dataset directory.

    Existing processed data is removed so that every execution starts
    from a clean and reproducible state.
    """

    print("\nPreparing output directories...")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        print(f"[OK] Previous processed dataset removed: {OUTPUT_DIR}")

    for split_name in SPLIT_NAMES:
        for class_name in SELECTED_CLASSES:
            destination = OUTPUT_DIR / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[OK] Processed dataset structure created: {OUTPUT_DIR}")
    print(f"[OK] Results directory ready: {RESULTS_DIR}")


def is_valid_image(image_path: Path) -> bool:
    """Return True if an image can be opened and verified successfully."""

    try:
        with Image.open(image_path) as image:
            image.verify()

        return True

    except (UnidentifiedImageError, OSError, ValueError):
        return False


def collect_valid_images(
    class_name: str,
) -> Tuple[List[Path], List[Path]]:
    """
    Collect valid image files for one class.

    Returns:
        A tuple containing:
        - valid image paths
        - invalid or corrupted image paths
    """

    class_directory = SOURCE_DIR / class_name

    candidate_files = sorted(
        file_path
        for file_path in class_directory.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    valid_images: List[Path] = []
    invalid_images: List[Path] = []

    for image_path in candidate_files:
        if is_valid_image(image_path):
            valid_images.append(image_path)
        else:
            invalid_images.append(image_path)

    if not valid_images:
        raise RuntimeError(
            f"No valid images were found for class '{class_name}'."
        )

    return valid_images, invalid_images


def split_images(
    image_paths: List[Path],
    random_generator: random.Random,
) -> Dict[str, List[Path]]:
    """
    Shuffle and divide image paths into train, validation, and test sets.
    """

    shuffled_images = image_paths.copy()
    random_generator.shuffle(shuffled_images)

    total_images = len(shuffled_images)

    train_count = int(total_images * TRAIN_RATIO)
    validation_count = int(total_images * VALIDATION_RATIO)

    train_end = train_count
    validation_end = train_count + validation_count

    return {
        "train": shuffled_images[:train_end],
        "validation": shuffled_images[train_end:validation_end],
        "test": shuffled_images[validation_end:],
    }


def copy_images(
    class_name: str,
    split_images_map: Dict[str, List[Path]],
) -> Dict[str, int]:
    """Copy one class into its train, validation, and test directories."""

    copied_counts: Dict[str, int] = {}

    for split_name, image_paths in split_images_map.items():
        destination_directory = OUTPUT_DIR / split_name / class_name

        for image_path in image_paths:
            destination_path = destination_directory / image_path.name
            shutil.copy2(image_path, destination_path)

        copied_counts[split_name] = len(image_paths)

    return copied_counts


def prepare_dataset() -> Tuple[Dict[str, Dict[str, int]], List[Path]]:
    """Prepare all selected classes and return processing statistics."""

    random_generator = random.Random(RANDOM_SEED)

    summary: Dict[str, Dict[str, int]] = {}
    all_invalid_images: List[Path] = []

    print("\nValidating, splitting, and copying images...")

    for class_name in SELECTED_CLASSES:
        print(f"\nProcessing class: {class_name}")

        valid_images, invalid_images = collect_valid_images(class_name)
        all_invalid_images.extend(invalid_images)

        print(f"  Valid images: {len(valid_images)}")
        print(f"  Invalid images: {len(invalid_images)}")

        image_splits = split_images(valid_images, random_generator)
        copied_counts = copy_images(class_name, image_splits)

        summary[class_name] = {
            "original": len(valid_images),
            "train": copied_counts["train"],
            "validation": copied_counts["validation"],
            "test": copied_counts["test"],
        }

        print(
            "  Split completed: "
            f"train={copied_counts['train']}, "
            f"validation={copied_counts['validation']}, "
            f"test={copied_counts['test']}"
        )

    return summary, all_invalid_images


def save_summary_csv(
    summary: Dict[str, Dict[str, int]],
) -> None:
    """Save dataset statistics as a CSV file."""

    with SUMMARY_CSV_PATH.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        fieldnames = [
            "class",
            "valid_images",
            "train_images",
            "validation_images",
            "test_images",
        ]

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for class_name in SELECTED_CLASSES:
            counts = summary[class_name]

            writer.writerow(
                {
                    "class": class_name,
                    "valid_images": counts["original"],
                    "train_images": counts["train"],
                    "validation_images": counts["validation"],
                    "test_images": counts["test"],
                }
            )


def save_summary_txt(
    summary: Dict[str, Dict[str, int]],
    invalid_images: List[Path],
) -> None:
    """Save a human-readable dataset report as a text file."""

    total_valid = sum(
        counts["original"] for counts in summary.values()
    )
    total_train = sum(
        counts["train"] for counts in summary.values()
    )
    total_validation = sum(
        counts["validation"] for counts in summary.values()
    )
    total_test = sum(
        counts["test"] for counts in summary.values()
    )

    report_lines = [
        "WASTE MATERIAL CLASSIFICATION",
        "DATASET PREPARATION SUMMARY",
        "=" * 50,
        "",
        f"Source directory: {SOURCE_DIR}",
        f"Processed directory: {OUTPUT_DIR}",
        f"Selected classes: {', '.join(SELECTED_CLASSES)}",
        f"Random seed: {RANDOM_SEED}",
        (
            "Split ratios: "
            f"train={TRAIN_RATIO:.0%}, "
            f"validation={VALIDATION_RATIO:.0%}, "
            f"test={TEST_RATIO:.0%}"
        ),
        "",
        "CLASS DISTRIBUTION",
        "-" * 50,
    ]

    for class_name in SELECTED_CLASSES:
        counts = summary[class_name]

        report_lines.extend(
            [
                f"Class: {class_name}",
                f"  Valid images: {counts['original']}",
                f"  Train images: {counts['train']}",
                f"  Validation images: {counts['validation']}",
                f"  Test images: {counts['test']}",
                "",
            ]
        )

    report_lines.extend(
        [
            "TOTALS",
            "-" * 50,
            f"Total valid images: {total_valid}",
            f"Total train images: {total_train}",
            f"Total validation images: {total_validation}",
            f"Total test images: {total_test}",
            f"Invalid images skipped: {len(invalid_images)}",
        ]
    )

    if invalid_images:
        report_lines.extend(
            [
                "",
                "INVALID IMAGE FILES",
                "-" * 50,
            ]
        )

        report_lines.extend(
            str(image_path) for image_path in invalid_images
        )

    SUMMARY_TXT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )


def print_summary(
    summary: Dict[str, Dict[str, int]],
    invalid_images: List[Path],
) -> None:
    """Print dataset statistics after preparation."""

    print("\n" + "=" * 72)
    print("DATASET PREPARATION SUMMARY")
    print("=" * 72)

    header = (
        f"{'Class':<15}"
        f"{'Valid':>10}"
        f"{'Train':>10}"
        f"{'Validation':>14}"
        f"{'Test':>10}"
    )

    print(header)
    print("-" * 72)

    total_valid = 0
    total_train = 0
    total_validation = 0
    total_test = 0

    for class_name in SELECTED_CLASSES:
        counts = summary[class_name]

        print(
            f"{class_name:<15}"
            f"{counts['original']:>10}"
            f"{counts['train']:>10}"
            f"{counts['validation']:>14}"
            f"{counts['test']:>10}"
        )

        total_valid += counts["original"]
        total_train += counts["train"]
        total_validation += counts["validation"]
        total_test += counts["test"]

    print("-" * 72)

    print(
        f"{'TOTAL':<15}"
        f"{total_valid:>10}"
        f"{total_train:>10}"
        f"{total_validation:>14}"
        f"{total_test:>10}"
    )

    print(f"\nSelected classes: {', '.join(SELECTED_CLASSES)}")
    print(f"Random seed: {RANDOM_SEED}")

    print(
        "Split ratios: "
        f"train={TRAIN_RATIO:.0%}, "
        f"validation={VALIDATION_RATIO:.0%}, "
        f"test={TEST_RATIO:.0%}"
    )

    print(f"Invalid images skipped: {len(invalid_images)}")
    print(f"Processed dataset location: {OUTPUT_DIR}")
    print(f"CSV summary saved to: {SUMMARY_CSV_PATH}")
    print(f"Text summary saved to: {SUMMARY_TXT_PATH}")

    if invalid_images:
        print("\nInvalid image files:")

        for image_path in invalid_images:
            print(f"  - {image_path}")

    print("\nDataset prepared successfully.")


def main() -> None:
    """Run the complete dataset preparation pipeline."""

    print("=" * 72)
    print("WASTE MATERIAL CLASSIFICATION - DATASET PREPARATION")
    print("=" * 72)

    check_configuration()
    reset_output_directory()

    summary, invalid_images = prepare_dataset()

    save_summary_csv(summary)
    save_summary_txt(summary, invalid_images)
    print_summary(summary, invalid_images)


if __name__ == "__main__":
    main()