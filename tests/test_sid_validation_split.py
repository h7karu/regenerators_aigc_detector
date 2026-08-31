import pandas as pd

from manifest_schema import normalise_manifest, validate_manifest
from scripts.split_sid_validation import stratified_split


def test_sid_validation_split_is_stratified_and_leak_free() -> None:
    rows = []
    for original_label, count in ((0, 8), (1, 4), (2, 4)):
        for index in range(count):
            rows.append(
                {
                    "path": f"image-{original_label}-{index}.png",
                    "label": int(original_label != 0),
                    "original_label": original_label,
                    "split": "val",
                    "dataset": "sid",
                    "generator_family": (
                        "real" if original_label == 0 else "synthetic"
                    ),
                    "content_id": f"content-{original_label}-{index}",
                }
            )
    frame = normalise_manifest(pd.DataFrame(rows))

    validation, holdout = stratified_split(
        frame,
        validation_fraction=0.5,
        seed=7,
    )

    assert validation.groupby("original_label").size().to_dict() == {0: 4, 1: 2, 2: 2}
    assert holdout.groupby("original_label").size().to_dict() == {0: 4, 1: 2, 2: 2}
    assert set(validation["content_id"]).isdisjoint(holdout["content_id"])
    validate_manifest(pd.concat([validation, holdout], ignore_index=True))
