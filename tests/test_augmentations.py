import numpy as np
import pytest

from augmentations import (
    ROBUSTNESS_TRANSFORMS,
    apply_robustness_transform,
    build_paired_train_transform,
)


@pytest.fixture
def sample_image() -> np.ndarray:
    y, x = np.mgrid[0:64, 0:80]
    return np.stack(
        [x * 3 % 256, y * 4 % 256, (x + y) * 2 % 256], axis=-1
    ).astype(np.uint8)


@pytest.mark.parametrize("transform_name", ROBUSTNESS_TRANSFORMS)
def test_robustness_transforms_preserve_shape_and_dtype(
    sample_image: np.ndarray, transform_name: str
) -> None:
    transformed = apply_robustness_transform(
        sample_image, transform_name, seed=123
    )
    assert transformed.shape == sample_image.shape
    assert transformed.dtype == np.uint8


def test_seeded_noise_is_deterministic(sample_image: np.ndarray) -> None:
    first = apply_robustness_transform(sample_image, "noise_0.05", seed=7)
    second = apply_robustness_transform(sample_image, "noise_0.05", seed=7)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, sample_image)


def test_unknown_transform_is_rejected(sample_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="Unknown transform"):
        apply_robustness_transform(sample_image, "unknown")


def test_paired_pipeline_returns_two_normalized_views(
    sample_image: np.ndarray,
) -> None:
    pipeline = build_paired_train_transform(224)
    pipeline.sample_degradation = lambda: "jpeg_30"
    result = pipeline(image=sample_image)
    assert result["clean_image"].shape == (3, 224, 224)
    assert result["degraded_image"].shape == (3, 224, 224)
    assert result["degradation"] == "jpeg_30"
    assert not np.array_equal(
        result["clean_image"].numpy(), result["degraded_image"].numpy()
    )
