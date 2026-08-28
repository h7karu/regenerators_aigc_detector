"""Manifest-backed datasets and balanced sampling utilities."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, WeightedRandomSampler

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

        transformed = self.transform(image=image)
        sample: dict[str, object] = {
            "label": torch.tensor(float(record["label"]), dtype=torch.float32),
            "path": str(path),
            "dataset": str(record["dataset"]),
            "group": str(record["generator"]),
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
