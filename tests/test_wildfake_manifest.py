from pathlib import Path

import pandas as pd

from scripts.build_wildfake_manifest import carve_validation, publisher_to_manifest


def test_publisher_metadata_maps_to_hierarchy(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        [
            {
                "Generator": "Diffusion_based",
                "Architecture": "SD",
                "Weight": "personalizedSD",
                "Category": "art",
                "IsAdvanced": 1,
                "IsFake": 1,
                "Image_path": "/publisher/Images/Diffusion_based/SD/a.png",
                "Num": 10,
            },
            {
                "Generator": "Real",
                "Architecture": "coco",
                "Weight": "None",
                "Category": "photo",
                "IsAdvanced": 0,
                "IsFake": 0,
                "Image_path": "/publisher/Images/Real/coco/b.jpg",
                "Num": 11,
            },
        ]
    )
    frame = publisher_to_manifest(metadata, split="train", images_root=tmp_path)
    assert frame["label"].tolist() == [1, 0]
    assert frame["generator_family"].tolist() == ["diffusion", "real"]
    assert frame["version"].tolist() == ["advanced", "unknown"]
    assert frame.loc[0, "path"].endswith("Diffusion_based\\SD\\a.png") or frame.loc[
        0, "path"
    ].endswith("Diffusion_based/SD/a.png")


def test_validation_is_carved_only_from_official_train() -> None:
    rows = []
    for generator in ("g1", "g2"):
        for index in range(10):
            rows.append(
                {
                    "path": f"{generator}_{index}.png",
                    "label": 1,
                    "split": "train",
                    "dataset": "wildfake",
                    "generator_family": "diffusion",
                    "generator": generator,
                    "content_id": f"{generator}_{index}",
                }
            )
    from manifest_schema import normalise_manifest

    training, validation = carve_validation(normalise_manifest(pd.DataFrame(rows)), 0.2, 7)
    assert len(training) == 16
    assert len(validation) == 4
    assert set(training["content_id"]).isdisjoint(validation["content_id"])
