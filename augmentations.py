"""Training and deterministic robustness transformations for AIGC detection."""

from __future__ import annotations

from io import BytesIO
from typing import Callable

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageEnhance


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

ROBUSTNESS_TRANSFORMS = (
    "clean",
    "jpeg_90",
    "jpeg_70",
    "jpeg_50",
    "jpeg_30",
    "blur_0.5",
    "blur_1.0",
    "blur_2.0",
    "resize_0.5",
    "resize_0.25",
    "noise_0.02",
    "noise_0.05",
    "noise_0.10",
    "color_0.2",
    "crop_0.8",
)

TRAINING_DEGRADATION_CATEGORIES = (
    "clean",
    "jpeg",
    "blur",
    "resize",
    "noise",
    "color",
    "crop",
)
TRAINING_DEGRADATION_PROBABILITIES = (0.15, 0.25, 0.15, 0.15, 0.15, 0.10, 0.05)
TRAINING_DEGRADATION_OPTIONS = {
    "clean": ("clean",),
    "jpeg": ("jpeg_90", "jpeg_70", "jpeg_50", "jpeg_30"),
    "blur": ("blur_0.5", "blur_1.0", "blur_2.0"),
    "resize": ("resize_0.5", "resize_0.25"),
    "noise": ("noise_0.02", "noise_0.05", "noise_0.10"),
    "color": ("color_0.2",),
    "crop": ("crop_0.8",),
}


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    kernel_size = max(3, int(round(sigma * 6)) | 1)
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma)


def _resize_round_trip(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    down_width = max(1, round(width * scale))
    down_height = max(1, round(height * scale))
    reduced = cv2.resize(
        image, (down_width, down_height), interpolation=cv2.INTER_AREA
    )
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)


def _gaussian_noise(
    image: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    image_float = image.astype(np.float32) / 255.0
    noise = rng.normal(0.0, sigma, image_float.shape).astype(np.float32)
    return np.round(np.clip(image_float + noise, 0.0, 1.0) * 255.0).astype(
        np.uint8
    )


def _color_jitter(
    image: np.ndarray, amount: float, rng: np.random.Generator
) -> np.ndarray:
    result = Image.fromarray(image)
    for enhancer in (
        ImageEnhance.Brightness,
        ImageEnhance.Contrast,
        ImageEnhance.Color,
    ):
        factor = float(rng.uniform(1.0 - amount, 1.0 + amount))
        result = enhancer(result).enhance(factor)
    return np.asarray(result, dtype=np.uint8)


def _center_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    height, width = image.shape[:2]
    crop_height = max(1, round(height * fraction))
    crop_width = max(1, round(width * fraction))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    cropped = image[top : top + crop_height, left : left + crop_width]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_CUBIC)


def apply_robustness_transform(
    image: np.ndarray,
    transform_name: str,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Apply one named benchmark transformation while preserving image size."""
    if transform_name not in ROBUSTNESS_TRANSFORMS:
        raise ValueError(
            f"Unknown transform {transform_name!r}. "
            f"Expected one of {ROBUSTNESS_TRANSFORMS}."
        )
    if transform_name == "clean":
        return image.copy()

    operation, value_text = transform_name.split("_", maxsplit=1)
    value = float(value_text)
    rng = np.random.default_rng(seed)

    if operation == "jpeg":
        return _jpeg(image, int(value))
    if operation == "blur":
        return _gaussian_blur(image, value)
    if operation == "resize":
        return _resize_round_trip(image, value)
    if operation == "noise":
        return _gaussian_noise(image, value, rng)
    if operation == "color":
        return _color_jitter(image, value, rng)
    if operation == "crop":
        return _center_crop(image, value)
    raise AssertionError(f"Unhandled transformation: {operation}")


class ImagePipeline:
    """Apply optional robustness degradation followed by model preprocessing."""

    def __init__(
        self,
        preprocessing: A.Compose,
        robustness_transform: str = "clean",
    ) -> None:
        self.preprocessing = preprocessing
        self.robustness_transform = robustness_transform

    def __call__(self, image: np.ndarray) -> dict[str, object]:
        transformed = apply_robustness_transform(
            image, self.robustness_transform
        )
        return self.preprocessing(image=transformed)


class TTAImagePipeline:
    """Create deterministic deployment views after an optional base degradation."""

    def __init__(
        self,
        preprocessing: A.Compose,
        view_transforms: tuple[str, ...],
        base_transform: str = "clean",
    ) -> None:
        self.preprocessing = preprocessing
        self.view_transforms = view_transforms
        self.base_transform = base_transform

    def __call__(self, image: np.ndarray) -> dict[str, object]:
        base_pixels = apply_robustness_transform(image, self.base_transform)
        views = [
            self.preprocessing(
                image=apply_robustness_transform(base_pixels, transform_name)
            )["image"]
            for transform_name in self.view_transforms
        ]
        return {"image": torch.stack(views)}


class PairedTrainPipeline:
    """Create clean/degraded views after applying identical random geometry."""

    def __init__(self, image_size: int) -> None:
        self.geometry = A.Compose(
            [
                A.RandomResizedCrop(
                    size=(image_size, image_size),
                    scale=(0.75, 1.0),
                    ratio=(0.8, 1.25),
                    p=1.0,
                ),
                A.HorizontalFlip(p=0.5),
            ]
        )
        self.normalization = A.Compose(
            [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    @staticmethod
    def sample_degradation() -> str:
        category = str(
            np.random.choice(
                TRAINING_DEGRADATION_CATEGORIES,
                p=TRAINING_DEGRADATION_PROBABILITIES,
            )
        )
        return str(np.random.choice(TRAINING_DEGRADATION_OPTIONS[category]))

    def __call__(self, image: np.ndarray) -> dict[str, object]:
        clean_pixels = self.geometry(image=image)["image"]
        degradation = self.sample_degradation()
        seed = int(np.random.randint(0, np.iinfo(np.int32).max))
        degraded_pixels = apply_robustness_transform(
            clean_pixels, degradation, seed=seed
        )
        clean_tensor = self.normalization(image=clean_pixels)["image"]
        degraded_tensor = self.normalization(image=degraded_pixels)["image"]
        return {
            "clean_image": clean_tensor,
            "degraded_image": degraded_tensor,
            "degradation": degradation,
        }


def build_train_transform(image_size: int, *, robust: bool = False) -> Callable:
    transforms: list[A.BasicTransform] = [
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.75, 1.0),
            ratio=(0.8, 1.25),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
            p=0.3,
        ),
    ]
    if robust:
        transforms.append(
            A.OneOf(
                [
                    A.ImageCompression(quality_range=(30, 95), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 9), sigma_limit=(0.5, 2.0), p=1.0),
                    A.Downscale(scale_range=(0.25, 0.75), p=1.0),
                    A.GaussNoise(std_range=(0.02, 0.10), p=1.0),
                ],
                p=0.7,
            )
        )
    transforms.extend(
        [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def build_eval_transform(
    image_size: int, robustness_transform: str = "clean"
) -> ImagePipeline:
    preprocessing = A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return ImagePipeline(preprocessing, robustness_transform)


def build_tta_eval_transform(
    image_size: int,
    view_transforms: tuple[str, ...],
    *,
    base_transform: str = "clean",
) -> TTAImagePipeline:
    preprocessing = A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
    return TTAImagePipeline(preprocessing, view_transforms, base_transform)


def build_paired_train_transform(image_size: int) -> PairedTrainPipeline:
    return PairedTrainPipeline(image_size)
