import pandas as pd
import pytest

from augmentations import build_eval_transform, build_paired_train_transform
from datasets import ManifestImageDataset, create_balanced_sampler


def test_cifake_manifest_loads_an_image() -> None:
    dataset = ManifestImageDataset(
        "data/manifests/cifake_val.csv",
        build_eval_transform(224),
        limit=4,
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 224, 224)
    assert sample["label"].ndim == 0
    assert sample["dataset"] == "cifake"


def test_balanced_sampler_matches_dataset_length() -> None:
    dataset = ManifestImageDataset(
        "data/manifests/cifake_val.csv",
        build_eval_transform(224),
        limit=100,
    )
    sampler = create_balanced_sampler(dataset)
    assert sampler.num_samples == len(dataset)


def test_manifest_dataset_supports_paired_views() -> None:
    transform = build_paired_train_transform(224)
    transform.sample_degradation = lambda: "blur_1.0"
    dataset = ManifestImageDataset(
        "data/manifests/cifake_train.csv",
        transform,
        limit=1,
    )
    sample = dataset[0]
    assert sample["clean_image"].shape == (3, 224, 224)
    assert sample["degraded_image"].shape == (3, 224, 224)
    assert sample["degradation"] == "blur_1.0"


def test_hierarchical_sampler_balances_labels_then_generators() -> None:
    dataset = object.__new__(ManifestImageDataset)
    dataset.frame = pd.DataFrame(
        {
            "label": [0] * 6 + [1] * 6,
            "dataset": ["a"] * 12,
            "generator": ["camera"] * 6 + ["g1"] * 2 + ["g2"] * 4,
        }
    )
    sampler = create_balanced_sampler(
        dataset,
        columns=("label", "dataset", "generator"),
        seed=7,
    )
    weights = pd.Series(list(sampler.weights), dtype=float)
    frame = dataset.frame.assign(weight=weights)

    label_mass = frame.groupby("label")["weight"].sum()
    fake_generator_mass = frame[frame["label"] == 1].groupby("generator")[
        "weight"
    ].sum()
    assert label_mass.iloc[0] == pytest.approx(label_mass.iloc[1])
    assert fake_generator_mass.iloc[0] == pytest.approx(fake_generator_mass.iloc[1])
