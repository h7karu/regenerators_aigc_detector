import pandas as pd
import pytest

from manifest_schema import (
    ManifestValidationError,
    normalise_manifest,
    validate_manifest,
)


def valid_rows() -> pd.DataFrame:
    return normalise_manifest(
        pd.DataFrame(
            [
                {
                    "path": "real.jpg",
                    "label": 0,
                    "split": "train",
                    "dataset": "example",
                    "generator_family": "real",
                    "content_id": "real-1",
                },
                {
                    "path": "fake.jpg",
                    "label": 1,
                    "split": "test",
                    "dataset": "example",
                    "generator_family": "diffusion",
                    "content_id": "fake-1",
                },
            ]
        )
    )


def test_normalise_manifest_adds_canonical_metadata() -> None:
    frame = valid_rows()
    assert frame.loc[0, "original_label"] == 0
    assert frame.loc[0, "generator"] == "unknown"
    validate_manifest(frame)


def test_content_identity_cannot_cross_splits() -> None:
    frame = valid_rows()
    frame.loc[1, "content_id"] = "real-1"
    with pytest.raises(ManifestValidationError, match="cross split"):
        validate_manifest(frame)


def test_reserved_dalle_advanced_is_rejected() -> None:
    frame = valid_rows()
    frame.loc[1, "architecture"] = "DALL-E"
    frame.loc[1, "version"] = "advanced"
    with pytest.raises(ManifestValidationError, match="DALL-E Advanced"):
        validate_manifest(frame)
