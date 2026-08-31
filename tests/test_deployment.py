from pathlib import Path

from deployment import (
    DEFAULT_INFERENCE_AGGREGATION,
    DEFAULT_INFERENCE_THRESHOLD,
    DEFAULT_INFERENCE_TRANSFORMS,
    DEFAULT_MODEL_CHECKPOINT,
    DEFAULT_MODEL_CONFIG,
    NEGATIVE_VERDICT,
    POSITIVE_VERDICT,
)
from utils import load_config


def test_default_checkpoint_matches_default_config_output() -> None:
    config = load_config(DEFAULT_MODEL_CONFIG)
    configured_checkpoint = (
        Path(config["output"]["checkpoint_dir"])
        / config["output"]["checkpoint_name"]
    )

    assert configured_checkpoint.as_posix() == DEFAULT_MODEL_CHECKPOINT


def test_default_inference_policy_is_the_validated_tta_policy() -> None:
    assert DEFAULT_INFERENCE_AGGREGATION == "trimmed_mean"
    assert DEFAULT_INFERENCE_THRESHOLD == 0.4856400298408816
    assert DEFAULT_INFERENCE_TRANSFORMS == (
        "clean",
        "jpeg_70",
        "blur_1.0",
        "resize_0.5",
        "crop_0.8",
    )
    assert POSITIVE_VERDICT == "AI Generated or Manipulated"
    assert NEGATIVE_VERDICT == "Real"
