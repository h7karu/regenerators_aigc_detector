"""Convert WildFake's official train/test metadata into canonical manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path, PurePosixPath

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from manifest_schema import (  # noqa: E402
    normalise_manifest,
    prohibited_demo_mask,
    validate_manifest,
    write_manifest,
)
from utils import resolve_project_path  # noqa: E402


PUBLISHER_COLUMNS = {
    "generator",
    "architecture",
    "weight",
    "category",
    "isadvanced",
    "isfake",
    "image_path",
    "num",
}


def parse_boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(int(value))
    normalised = str(value).strip().lower()
    if normalised in {"1", "true", "yes", "y"}:
        return True
    if normalised in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot interpret boolean metadata value: {value!r}")


def canonical_publisher_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {column: column.strip().lower() for column in frame.columns}
    result = frame.rename(columns=renamed)
    missing = PUBLISHER_COLUMNS - set(result.columns)
    if missing:
        raise ValueError(f"WildFake metadata is missing columns: {sorted(missing)}")
    return result


def relocate_image_path(raw_path: object, images_root: Path) -> Path:
    """Map the publisher's original absolute path into an extracted Images tree."""

    posix = PurePosixPath(str(raw_path).replace("\\", "/"))
    parts = [part for part in posix.parts if part not in {"/", "\\"}]
    candidates: list[Path] = []

    if Path(str(raw_path)).is_absolute():
        candidates.append(Path(str(raw_path)))
    candidates.append(images_root.joinpath(*parts))

    lowered = [part.lower() for part in parts]
    for marker in ("images", "real", "diffusion_based", "gan_based", "other_based"):
        if marker in lowered:
            index = lowered.index(marker)
            suffix = parts[index + 1 :] if marker == "images" else parts[index:]
            candidates.append(images_root.joinpath(*suffix))

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[-1]


def publisher_to_manifest(
    metadata: pd.DataFrame,
    *,
    split: str,
    images_root: Path,
) -> pd.DataFrame:
    metadata = canonical_publisher_columns(metadata)
    rows: list[dict[str, object]] = []
    for record in metadata.to_dict(orient="records"):
        is_fake = parse_boolean(record["isfake"])
        is_advanced = parse_boolean(record["isadvanced"])
        family = str(record["generator"]).strip()
        architecture = str(record["architecture"]).strip()
        image_path = relocate_image_path(record["image_path"], images_root)
        try:
            stored_path = image_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            stored_path = str(image_path)
        version = "advanced" if is_advanced else "typical"
        if not is_fake:
            version = "unknown"
        rows.append(
            {
                "path": stored_path,
                "label": int(is_fake),
                "original_label": int(is_fake),
                "split": split,
                "dataset": "wildfake",
                "generator_family": family,
                "generator": architecture,
                "architecture": architecture,
                "weight_type": record["weight"],
                "version": version,
                "source": record["category"],
                "content_id": f"wildfake:{family}:{architecture}:{record['num']}",
                "publisher_path": record["image_path"],
                "publisher_index": record["num"],
            }
        )
    return normalise_manifest(pd.DataFrame(rows))


def carve_validation(
    train_frame: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("Validation fraction must be in [0, 1).")
    if fraction == 0.0:
        return train_frame.copy(), train_frame.iloc[:0].copy()

    validation_indices: list[int] = []
    grouping = ["label", "generator_family", "generator"]
    for _, group in train_frame.groupby(grouping, dropna=False, sort=True):
        count = min(max(round(len(group) * fraction), 1), max(len(group) - 1, 0))
        if count:
            validation_indices.extend(
                group.sample(n=count, random_state=seed).index.tolist()
            )
    validation = train_frame.loc[validation_indices].copy()
    training = train_frame.drop(index=validation_indices).copy()
    training["split"] = "train"
    validation["split"] = "val"
    return training, validation


def limit_groups(frame: pd.DataFrame, maximum: int | None, seed: int) -> pd.DataFrame:
    if maximum is None:
        return frame
    if maximum <= 0:
        raise ValueError("--max-per-group must be positive.")
    grouping = ["split", "label", "generator_family", "generator"]
    return (
        frame.groupby(grouping, dropna=False, group_keys=False, sort=True)
        .apply(lambda group: group.sample(n=min(len(group), maximum), random_state=seed))
        .reset_index(drop=True)
    )


def remove_prohibited_sources(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    prohibited = prohibited_demo_mask(frame)
    return frame.loc[~prohibited].copy(), int(prohibited.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--test-metadata", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output-dir", default="data/manifests")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-per-group", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Fail unless every referenced extracted image exists.",
    )
    args = parser.parse_args()

    images_root = resolve_project_path(args.images_root)
    train_metadata = pd.read_csv(resolve_project_path(args.train_metadata))
    test_metadata = pd.read_csv(resolve_project_path(args.test_metadata))
    official_train = publisher_to_manifest(
        train_metadata, split="train", images_root=images_root
    )
    test = publisher_to_manifest(test_metadata, split="test", images_root=images_root)
    train, validation = carve_validation(official_train, args.val_fraction, args.seed)

    combined = pd.concat([train, validation, test], ignore_index=True)
    combined, excluded_count = remove_prohibited_sources(combined)
    combined = limit_groups(combined, args.max_per_group, args.seed)
    validate_manifest(combined, check_paths=args.check_paths)

    output_directory = resolve_project_path(args.output_dir)
    for split in ("train", "val", "test"):
        split_frame = combined[combined["split"] == split].reset_index(drop=True)
        if not split_frame.empty:
            write_manifest(
                split_frame,
                output_directory / f"wildfake_{split}.csv",
                check_paths=args.check_paths,
            )
    write_manifest(
        combined,
        output_directory / "wildfake_all.csv",
        check_paths=args.check_paths,
    )

    print(f"Wrote {len(combined):,} WildFake records to {output_directory}")
    print(combined.groupby(["split", "generator_family", "generator"]).size())
    print(f"Excluded {excluded_count:,} reserved demonstration records")


if __name__ == "__main__":
    main()
