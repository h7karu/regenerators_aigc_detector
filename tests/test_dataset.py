import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from augmentations import build_eval_transform, build_paired_train_transform
import data_pipeline
from data_pipeline import (
    ManifestImageDataset,
    StreamingSIDDataset,
    create_balanced_sampler,
)


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


class FakeStream:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = ["img_id", "image", "mask", "label"]

    def remove_columns(self, columns):
        self.column_names = [name for name in self.column_names if name not in columns]
        return self

    def cast_column(self, *_args):
        return self

    def shuffle(self, **_kwargs):
        return self

    def shard(self, **_kwargs):
        return self

    def __iter__(self):
        return iter(self.rows)


def encoded_image(value: int) -> dict[str, object]:
    buffer = io.BytesIO()
    Image.fromarray(np.full((12, 12, 3), value, dtype=np.uint8)).save(
        buffer, format="PNG"
    )
    return {"bytes": buffer.getvalue(), "path": None}


def test_sid_stream_is_binary_balanced_and_drops_masks(monkeypatch) -> None:
    labels = [1, 2, 0, 1, 0, 2, 0, 0]
    stream = FakeStream(
        [
            {
                "img_id": f"image-{index}",
                "image": encoded_image(index),
                "mask": encoded_image(0),
                "label": label,
            }
            for index, label in enumerate(labels)
        ]
    )
    monkeypatch.setattr(data_pipeline, "load_dataset", lambda *args, **kwargs: stream)
    dataset = StreamingSIDDataset(
        build_eval_transform(16),
        split="train",
        samples_per_epoch=8,
        shuffle_buffer=2,
    )

    samples = list(dataset)

    assert [int(sample["label"]) for sample in samples] == [0, 1, 0, 1] * 2
    assert [sample["group"] for sample in samples] == [
        "real",
        "full_synthetic",
        "real",
        "tampered",
    ] * 2
    assert "mask" not in stream.column_names
    assert all(sample["image"].shape == (3, 16, 16) for sample in samples)


def test_sid_stream_epoch_changes_shuffle_seed(monkeypatch) -> None:
    calls = []
    stream = FakeStream([])

    def fake_load(*args, **kwargs):
        original_shuffle = stream.shuffle

        def record_shuffle(**shuffle_kwargs):
            calls.append(shuffle_kwargs)
            return original_shuffle(**shuffle_kwargs)

        stream.shuffle = record_shuffle
        return stream

    monkeypatch.setattr(data_pipeline, "load_dataset", fake_load)
    dataset = StreamingSIDDataset(
        build_eval_transform(16),
        split="train",
        samples_per_epoch=1,
        shuffle_buffer=32,
        seed=40,
    )
    dataset.set_epoch(2)
    dataset._stream()

    assert calls[-1] == {"seed": 42, "buffer_size": 32}
