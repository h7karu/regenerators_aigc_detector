"""Dataset loaders: labeled CIFAKE splits for train/eval, unlabeled dirs for inference."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset

from aigc_detector.utils import list_images

# CIFAKE label convention used throughout this project: 0 = REAL, 1 = FAKE (AIGC).
LABEL_NAMES = {0: "REAL", 1: "FAKE"}


class CIFAKEDataset(Dataset):
    """Reads a CIFAKE-style split with REAL/ and FAKE/ subfolders.

    `split_dir` should be e.g. `data/cifake/train` or `data/cifake/test`.
    """

    def __init__(
        self,
        split_dir: str | Path,
        transform: Callable[[Image.Image], Image.Image] | None = None,
        max_per_class: int | None = None,
        seed: int = 42,
    ):
        split_dir = Path(split_dir)
        real_paths = list_images(split_dir / "REAL")
        fake_paths = list_images(split_dir / "FAKE")
        if max_per_class is not None:
            # A plain paths[:max_per_class] silently starves whichever source
            # sorts last: our multi-source folders (e.g. WildFake's
            # ddim_/ddpm_/vqdm_ prefixes) are alphabetically grouped by source,
            # so capping the sorted list can drop an entire generator family
            # from training rather than sampling evenly across all of them.
            # Shuffle deterministically first so the cap draws from everywhere.
            real_paths = random.Random(seed).sample(real_paths, min(max_per_class, len(real_paths)))
            fake_paths = random.Random(seed + 1).sample(fake_paths, min(max_per_class, len(fake_paths)))

        self.samples: list[tuple[Path, int]] = [(p, 0) for p in real_paths] + [
            (p, 1) for p in fake_paths
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(path)


class ImageDirDataset(Dataset):
    """Unlabeled dataset for inference: every image file found under `root` (recursive)."""

    def __init__(
        self,
        root: str | Path,
        transform: Callable[[Image.Image], Image.Image] | None = None,
    ):
        self.paths = list_images(Path(root))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(path)


def collate_keep_pil(batch):
    """Collate that keeps PIL images as a plain list (CLIP processor batches them itself)."""
    return list(zip(*batch))
