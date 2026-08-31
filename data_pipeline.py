"""Manifest-backed and Hugging Face streaming data pipelines."""

from __future__ import annotations

import io
from collections import deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from datasets import Image as HuggingFaceImage
from datasets import load_dataset
from PIL import Image, ImageOps
from torch.utils.data import Dataset, IterableDataset, WeightedRandomSampler

from manifest_schema import normalise_manifest, validate_manifest
from utils import PROJECT_ROOT, resolve_project_path


class ManifestImageDataset(Dataset):
    REQUIRED_COLUMNS = {"path", "label", "split", "dataset"}

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable,
        *,
        limit: int | None = None,
    ) -> None:
        self.manifest_path = resolve_project_path(manifest_path)
        self.frame = normalise_manifest(pd.read_csv(self.manifest_path))
        validate_manifest(self.frame, forbid_demo_sources=True)
        if limit is not None:
            self.frame = self.frame.iloc[:limit].copy()
        self.frame["label"] = self.frame["label"].astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.frame.iloc[index]
        path = Path(record["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            with Image.open(path) as source:
                image = np.asarray(
                    ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8
                )
        except Exception as error:
            raise RuntimeError(f"Unable to decode image: {path}") from error

        return transformed_sample(
            image,
            float(record["label"]),
            self.transform,
            path=str(path),
            dataset=str(record["dataset"]),
            group=str(record["generator"]),
        )


def transformed_sample(
    image: np.ndarray,
    label: float,
    transform: Callable,
    *,
    path: str,
    dataset: str,
    group: str,
) -> dict[str, object]:
    transformed = transform(image=image)
    sample: dict[str, object] = {
        "label": torch.tensor(label, dtype=torch.float32),
        "path": path,
        "dataset": dataset,
        "group": group,
    }
    if "image" in transformed:
        sample["image"] = transformed["image"]
    elif "clean_image" in transformed and "degraded_image" in transformed:
        sample["clean_image"] = transformed["clean_image"]
        sample["degraded_image"] = transformed["degraded_image"]
        sample["degradation"] = transformed["degradation"]
    else:
        raise ValueError(
            "Transform must return either image or clean/degraded image keys."
        )
    return sample


def decode_streamed_image(value: object) -> np.ndarray:
    """Decode a non-cached Hugging Face Image value into RGB pixels."""

    if isinstance(value, Image.Image):
        return np.asarray(ImageOps.exif_transpose(value).convert("RGB"), dtype=np.uint8)
    if not isinstance(value, dict):
        raise TypeError(f"Unsupported streamed image value: {type(value).__name__}")
    encoded = value.get("bytes")
    if encoded is not None:
        source = io.BytesIO(encoded)
    elif value.get("path"):
        source = value["path"]
    else:
        raise ValueError("Streamed image has neither bytes nor a path.")
    with Image.open(source) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


class StreamingSIDDataset(IterableDataset):
    """Stream a reproducible, binary-balanced SID-Set epoch without local images."""

    BALANCE_PATTERN = (0, 1, 0, 2)
    GROUP_NAMES = {0: "real", 1: "full_synthetic", 2: "tampered"}

    def __init__(
        self,
        transform: Callable,
        *,
        split: str,
        samples_per_epoch: int,
        dataset_id: str = "saberzl/SID_Set",
        shuffle_buffer: int = 10_000,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("SID split must be 'train' or 'validation'.")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive.")
        if shuffle_buffer < 1:
            raise ValueError("shuffle_buffer must be positive.")
        self.transform = transform
        self.split = split
        self.samples_per_epoch = samples_per_epoch
        self.dataset_id = dataset_id
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _stream(self):
        stream = load_dataset(self.dataset_id, split=self.split, streaming=True)
        keep = {"img_id", "image", "label"}
        removable = [name for name in stream.column_names if name not in keep]
        if removable:
            stream = stream.remove_columns(removable)
        stream = stream.cast_column("image", HuggingFaceImage(decode=False))
        if self.split == "train" and self.shuffle_buffer > 1:
            stream = stream.shuffle(
                seed=self.seed + self.epoch,
                buffer_size=self.shuffle_buffer,
            )
        return stream

    def __iter__(self) -> Iterator[dict[str, object]]:
        worker = torch.utils.data.get_worker_info()
        stream = self._stream()
        target_samples = self.samples_per_epoch
        if worker is not None:
            stream = stream.shard(num_shards=worker.num_workers, index=worker.id)
            quotient, remainder = divmod(self.samples_per_epoch, worker.num_workers)
            target_samples = quotient + int(worker.id < remainder)

        # Bound encoded-image memory even if the remote rows are ordered by class.
        queues = {label: deque(maxlen=16) for label in self.GROUP_NAMES}
        pattern_index = 0
        yielded = 0
        for row in stream:
            original_label = int(row["label"])
            if original_label not in queues:
                raise ValueError(f"Unknown SID label: {original_label}")
            queues[original_label].append(row)

            wanted = self.BALANCE_PATTERN[pattern_index]
            while queues[wanted]:
                selected = queues[wanted].popleft()
                pixels = decode_streamed_image(selected["image"])
                selected_label = int(selected["label"])
                yield transformed_sample(
                    pixels,
                    float(selected_label != 0),
                    self.transform,
                    path=f"hf://{self.dataset_id}/{self.split}/{selected['img_id']}",
                    dataset="sid",
                    group=self.GROUP_NAMES[selected_label],
                )
                yielded += 1
                if yielded >= target_samples:
                    return
                pattern_index = (pattern_index + 1) % len(self.BALANCE_PATTERN)
                wanted = self.BALANCE_PATTERN[pattern_index]

        raise RuntimeError(
            f"SID stream ended after {yielded:,} balanced samples; "
            f"{target_samples:,} were requested by this worker."
        )


def create_balanced_sampler(
    dataset: ManifestImageDataset,
    columns: Sequence[str] = ("label",),
    *,
    seed: int | None = None,
) -> WeightedRandomSampler:
    """Balance each level conditionally (label, then dataset, then generator)."""

    missing = set(columns) - set(dataset.frame.columns)
    if missing:
        raise ValueError(f"Cannot balance missing manifest columns: {sorted(missing)}")
    if not columns:
        raise ValueError("At least one balancing column is required.")

    metadata = dataset.frame[list(columns)].astype(str)
    sample_weights = np.ones(len(metadata), dtype=np.float64)
    prefix: list[str] = []
    for column in columns:
        if prefix:
            branch_counts = metadata.groupby(prefix, dropna=False)[column].transform(
                "nunique"
            )
        else:
            branch_counts = pd.Series(metadata[column].nunique(), index=metadata.index)
        sample_weights /= branch_counts.to_numpy(dtype=np.float64)
        prefix.append(column)

    leaf_counts = metadata.groupby(list(columns), dropna=False)[columns[0]].transform(
        "size"
    )
    sample_weights /= leaf_counts.to_numpy(dtype=np.float64)
    weights = torch.as_tensor(sample_weights, dtype=torch.double)
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
