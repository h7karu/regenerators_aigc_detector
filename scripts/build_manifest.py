"""Create portable CIFAKE CSV manifests and optionally audit file hashes."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import resolve_project_path  # noqa: E402
from manifest_schema import write_manifest  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
LABELS = {"REAL": 0, "FAKE": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_cifake(
    root: Path, include_hash: bool, verify_images: bool
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ("train", "test"):
        for class_name, label in LABELS.items():
            class_directory = root / split / class_name
            if not class_directory.is_dir():
                raise FileNotFoundError(f"Missing CIFAKE directory: {class_directory}")
            for path in sorted(class_directory.iterdir()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                width: int | None = None
                height: int | None = None
                if verify_images:
                    try:
                        with Image.open(path) as image:
                            width, height = image.size
                            image.verify()
                    except Exception as error:
                        raise RuntimeError(f"Corrupt image encountered: {path}") from error
                relative_path = path.relative_to(PROJECT_ROOT).as_posix()
                row: dict[str, object] = {
                    "path": relative_path,
                    "label": label,
                    "original_label": label,
                    "split": split,
                    "dataset": "cifake",
                    "generator_family": "diffusion" if label else "real",
                    "generator": "stable_diffusion_1_4" if label else "cifar10",
                    "architecture": "stable_diffusion" if label else "unknown",
                    "weight_type": "original" if label else "unknown",
                    "version": "1.4" if label else "unknown",
                    "source": "cifake_sd14" if label else "cifar10",
                    "content_id": relative_path,
                    "width": width,
                    "height": height,
                    "file_format": path.suffix.lower().lstrip("."),
                }
                if include_hash:
                    row["sha256"] = sha256(path)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="cifake-real-and-ai-generated-synthetic-images",
        help="CIFAKE directory, relative to the repository root by default.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/manifests",
        help="Directory in which train/test/all manifests are written.",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Compute SHA-256 hashes for duplicate auditing (slower).",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Decode every image during manifest creation (slower).",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = resolve_project_path(args.root)
    output_directory = resolve_project_path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    frame = scan_cifake(root, args.hash, args.verify_images)
    original_train = frame[frame["split"] == "train"].copy()
    test_frame = frame[frame["split"] == "test"].copy()
    train_frame, validation_frame = train_test_split(
        original_train,
        test_size=args.val_fraction,
        random_state=args.seed,
        stratify=original_train["label"],
    )
    train_frame = train_frame.copy()
    validation_frame = validation_frame.copy()
    train_frame["split"] = "train"
    validation_frame["split"] = "val"

    final_frame = pd.concat(
        [train_frame, validation_frame, test_frame], ignore_index=True
    )
    write_manifest(final_frame, output_directory / "cifake_all.csv", check_paths=True)
    for split, split_frame in (
        ("train", train_frame),
        ("val", validation_frame),
        ("test", test_frame),
    ):
        write_manifest(
            split_frame.reset_index(drop=True),
            output_directory / f"cifake_{split}.csv",
            check_paths=True,
        )

    counts = final_frame.groupby(["split", "label"]).size()
    print(f"Wrote {len(final_frame):,} records to {output_directory}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
