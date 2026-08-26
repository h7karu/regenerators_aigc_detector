"""Shared helpers for normalizing any source dataset into the CIFAKE layout.

Every dataset we use gets written out as:

    data/<name>/<split>/REAL/*.<ext>
    data/<name>/<split>/FAKE/*.<ext>

so a single loader (`CIFAKEDataset`) works across CIFAKE, SID_Set and WildFake
and the datasets can be mixed freely during training.
"""
from __future__ import annotations

import io
import random
from pathlib import Path

LABEL_DIRS = {0: "REAL", 1: "FAKE"}

# Defaults for label-leak normalization (see `normalize_image_bytes`).
NORMALIZE_MAX_DIM = 768
NORMALIZE_QUALITY = 95


def normalize_image_bytes(
    payload: bytes,
    max_dim: int = NORMALIZE_MAX_DIM,
    quality: int = NORMALIZE_QUALITY,
) -> tuple[bytes, str]:
    """Re-encode an image so container format and scale can't leak the label.

    These datasets ship the two classes in systematically different containers:
    in SID_Set the real images are JPEGs at assorted sizes while the synthetic
    ones are 1024x1024 PNGs; WildFake is similar (JPEG photos vs PNG samples).
    A classifier can then hit near-perfect accuracy by keying on "PNG and
    square" and learn nothing about generation at all -- and that shortcut
    evaporates the moment it meets a real image that happens to be a PNG.

    Writing every image through the same encoder at the same quality and the
    same size cap removes that shortcut, so what remains for the model to use is
    the actual generative signal.

    Trade-off: re-encoding does attenuate some of the very high-frequency detail
    the forensic branch reads. It is applied identically to both classes, so it
    costs a little signal rather than creating a bias -- a trade worth making,
    since a shortcut-driven model would post great offline numbers and then fail
    on real traffic. Pass `--no-normalize` to the download scripts to keep the
    original bytes and measure the difference.
    """
    from PIL import Image

    image = Image.open(io.BytesIO(payload))
    image = image.convert("RGB")

    if max_dim and max(image.size) > max_dim:
        scale = max_dim / max(image.size)
        new_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(new_size, Image.BICUBIC)

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue(), ".jpg"


def split_counts(total: int, test_fraction: float) -> tuple[int, int]:
    """Split `total` into (n_train, n_test), keeping at least 1 test item when possible."""
    n_test = int(round(total * test_fraction))
    n_test = max(1, min(n_test, total - 1)) if total > 1 else 0
    return total - n_test, n_test


def sample_deterministic(items: list, k: int, seed: int) -> list:
    """Pick `k` items reproducibly (so teammates running the script get the same subset)."""
    if k >= len(items):
        return list(items)
    return random.Random(seed).sample(sorted(items), k)


def target_dir(root: Path, dataset: str, split: str, label: int) -> Path:
    return Path(root) / dataset / split / LABEL_DIRS[label]


def write_bytes(dest_dir: Path, filename: str, payload: bytes) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    path.write_bytes(payload)
    return path


def summarize(root: Path, dataset: str) -> None:
    """Print the resulting per-split/per-class counts."""
    base = Path(root) / dataset
    print(f"\nContents of {base}:")
    for split_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for label_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            n = sum(1 for p in label_dir.iterdir() if p.is_file())
            print(f"  {split_dir.name}/{label_dir.name}: {n} images")
