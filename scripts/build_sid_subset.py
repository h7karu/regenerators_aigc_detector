"""Stream and materialise a compact, class-controlled SID-Set subset."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from datasets import Image as HuggingFaceImage
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from manifest_schema import write_manifest  # noqa: E402
from utils import resolve_project_path  # noqa: E402


SID_TYPES = {
    0: ("real", "openimages_v7"),
    1: ("synthetic", "full_synthetic"),
    2: ("tampered", "tampered"),
}


def image_extension(image: Image.Image) -> str:
    image_format = (image.format or "png").lower()
    return {"jpeg": "jpg", "tiff": "tif"}.get(image_format, image_format)


def materialise_image(value: object, destination_stem: Path) -> tuple[Path, int, int]:
    """Preserve encoded bytes when available; otherwise save losslessly as PNG."""

    encoded: bytes | None = None
    if isinstance(value, dict):
        encoded_value = value.get("bytes")
        if isinstance(encoded_value, bytes):
            encoded = encoded_value
        elif value.get("path"):
            encoded = Path(str(value["path"])).read_bytes()

    if encoded is not None:
        with Image.open(io.BytesIO(encoded)) as image:
            width, height = image.size
            extension = image_extension(image)
        destination = destination_stem.with_suffix(f".{extension}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(encoded)
        temporary.replace(destination)
        return destination, width, height

    if not isinstance(value, Image.Image):
        raise TypeError(f"Unsupported SID image value: {type(value).__name__}")
    image = value.convert("RGB")
    width, height = image.size
    destination = destination_stem.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".png.tmp")
    image.save(temporary, format="PNG")
    temporary.replace(destination)
    return destination, width, height


def binary_label(original_label: int) -> int:
    if original_label not in SID_TYPES:
        raise ValueError(f"Unknown SID label: {original_label}")
    return int(original_label != 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", default="saberzl/SID_Set")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--real-count", type=int, default=1000)
    parser.add_argument("--synthetic-count", type=int, default=500)
    parser.add_argument("--tampered-count", type=int, default=500)
    parser.add_argument("--output-dir", default="data/sid_subset")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--shuffle-buffer", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    quotas = {
        0: args.real_count,
        1: args.synthetic_count,
        2: args.tampered_count,
    }
    if any(count < 0 for count in quotas.values()) or not any(quotas.values()):
        raise ValueError("SID quotas must be non-negative and at least one must be positive.")

    stream = load_dataset(args.dataset_id, split=args.split, streaming=True)
    keep = {"img_id", "image", "label"}
    removable = [name for name in stream.column_names if name not in keep]
    if removable:
        stream = stream.remove_columns(removable)
    stream = stream.cast_column("image", HuggingFaceImage(decode=False))
    if args.shuffle_buffer > 1:
        stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    output_directory = resolve_project_path(args.output_dir)
    split = "val" if args.split == "validation" else "train"
    counts = {label: 0 for label in SID_TYPES}
    records: list[dict[str, object]] = []
    progress = tqdm(total=sum(quotas.values()), unit="image")
    for row in stream:
        original_label = int(row["label"])
        if original_label not in quotas or counts[original_label] >= quotas[original_label]:
            continue

        family, source = SID_TYPES[original_label]
        image_id = str(row["img_id"])
        safe_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in image_id
        )
        destination, width, height = materialise_image(
            row["image"],
            output_directory / split / source / safe_id,
        )
        try:
            stored_path = destination.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            stored_path = str(destination)

        records.append(
            {
                "path": stored_path,
                "label": binary_label(original_label),
                "original_label": original_label,
                "split": split,
                "dataset": "sid",
                "generator_family": family,
                "generator": source,
                "architecture": "unknown",
                "weight_type": "unknown",
                "version": "unknown",
                "source": source,
                # Synthetic/tampered img_id values are only unique within the
                # published HF split, so include that namespace for leakage
                # checks across materialised train/validation manifests.
                "content_id": f"{args.split}:{image_id}",
                "width": width,
                "height": height,
                "file_format": destination.suffix.lower().lstrip("."),
            }
        )
        counts[original_label] += 1
        progress.update(1)
        if counts == quotas:
            break
    progress.close()

    if counts != quotas:
        raise RuntimeError(f"SID stream ended before quotas were filled: {counts} != {quotas}")
    manifest_path = (
        resolve_project_path(args.manifest_dir) / f"sid_{split}.csv"
    )
    frame = write_manifest(records, manifest_path, check_paths=True)
    print(f"Wrote {len(frame):,} SID images and {manifest_path}")
    print(frame.groupby(["split", "original_label", "generator"]).size())


if __name__ == "__main__":
    main()
