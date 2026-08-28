"""Canonical manifest schema and leakage checks for detector datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from utils import PROJECT_ROOT


MANIFEST_COLUMNS = (
    "path",
    "label",
    "original_label",
    "split",
    "dataset",
    "generator_family",
    "generator",
    "architecture",
    "weight_type",
    "version",
    "source",
    "content_id",
    "width",
    "height",
    "file_format",
    "sha256",
)

REQUIRED_COLUMNS = {"path", "label", "split", "dataset"}
VALID_SPLITS = {"train", "val", "validation", "test"}
UNKNOWN = "unknown"


class ManifestValidationError(ValueError):
    """Raised when a manifest can leak data or violates the data contract."""


def _clean_text(series: pd.Series, default: str = UNKNOWN) -> pd.Series:
    cleaned = series.fillna(default).astype(str).str.strip()
    return cleaned.mask(cleaned.eq(""), default)


def normalise_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical copy while retaining any dataset-specific columns."""

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ManifestValidationError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    result = frame.copy()
    result["label"] = pd.to_numeric(result["label"], errors="raise").astype(int)
    if "original_label" not in result:
        result["original_label"] = result["label"]

    text_defaults = {
        "path": "",
        "split": "",
        "dataset": "",
        "generator_family": UNKNOWN,
        "generator": UNKNOWN,
        "architecture": UNKNOWN,
        "weight_type": UNKNOWN,
        "version": UNKNOWN,
        "source": UNKNOWN,
        "content_id": UNKNOWN,
        "file_format": UNKNOWN,
        "sha256": UNKNOWN,
    }
    for column, default in text_defaults.items():
        if column not in result:
            result[column] = default
        result[column] = _clean_text(result[column], default)

    result["split"] = result["split"].str.lower().replace("validation", "val")
    result["dataset"] = result["dataset"].str.lower()
    result["generator_family"] = (
        result["generator_family"]
        .str.lower()
        .replace(
            {
                "real_based": "real",
                "gan_based": "gan",
                "diffusion_based": "diffusion",
                "other_based": "other",
            }
        )
    )
    for column in ("width", "height"):
        if column not in result:
            result[column] = pd.NA

    leading = list(MANIFEST_COLUMNS)
    trailing = [column for column in result.columns if column not in leading]
    return result[leading + trailing]


def prohibited_demo_reason(record: pd.Series) -> str | None:
    """Identify hackathon demonstration sources that must never enter training."""

    searchable = " ".join(
        str(record.get(column, ""))
        for column in (
            "path",
            "source",
            "generator",
            "architecture",
            "version",
        )
    ).lower()
    compact = "".join(character for character in searchable if character.isalnum())
    if "coco" in searchable and "val2017" in compact:
        return "COCO val2017 is reserved for the hackathon demonstration set"

    architecture = str(record.get("architecture", "")).lower()
    generator = str(record.get("generator", "")).lower()
    version = str(record.get("version", "")).lower()
    is_dalle = "dalle" in "".join(
        character for character in architecture + generator if character.isalnum()
    )
    if is_dalle and version in {"advanced", "dalle3", "3", "true", "1"}:
        return "DALL-E Advanced is reserved for the hackathon demonstration set"
    return None


def prohibited_demo_mask(frame: pd.DataFrame) -> pd.Series:
    """Vectorised counterpart used for million-row dataset manifests."""

    searchable_columns = [
        column
        for column in ("path", "source", "generator", "architecture", "version")
        if column in frame
    ]
    searchable = (
        frame[searchable_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )
    compact = searchable.str.replace(r"[^a-z0-9]", "", regex=True)
    coco_val2017 = searchable.str.contains("coco", regex=False) & compact.str.contains(
        "val2017", regex=False
    )

    architecture = frame.get("architecture", pd.Series("", index=frame.index)).astype(str)
    generator = frame.get("generator", pd.Series("", index=frame.index)).astype(str)
    dalle = (
        architecture.str.cat(generator)
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
        .str.contains("dalle", regex=False)
    )
    version = (
        frame.get("version", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    dalle_advanced = dalle & version.isin({"advanced", "dalle3", "3", "true", "1"})
    return coco_val2017 | dalle_advanced


def validate_manifest(
    frame: pd.DataFrame,
    *,
    check_paths: bool = False,
    forbid_demo_sources: bool = True,
    project_root: Path = PROJECT_ROOT,
) -> None:
    """Validate labels, splits, duplicates, leakage, and optional local paths."""

    errors: list[str] = []
    if frame.empty:
        errors.append("manifest has no records")

    invalid_labels = sorted(set(frame["label"]) - {0, 1})
    if invalid_labels:
        errors.append(f"binary label must be 0 or 1; found {invalid_labels}")

    invalid_splits = sorted(set(frame["split"]) - {"train", "val", "test"})
    if invalid_splits:
        errors.append(f"invalid split values: {invalid_splits}")

    empty_paths = frame["path"].astype(str).str.strip().eq("")
    if empty_paths.any():
        errors.append(f"{int(empty_paths.sum())} records have empty paths")

    duplicate_paths = frame[frame["path"].duplicated(keep=False)]["path"].unique()
    if len(duplicate_paths):
        errors.append(
            f"duplicate image paths found (first: {str(duplicate_paths[0])!r})"
        )

    known_content = frame[~frame["content_id"].isin({"", UNKNOWN})]
    if not known_content.empty:
        split_counts = known_content.groupby("content_id")["split"].nunique()
        leaking_ids = split_counts[split_counts > 1]
        if not leaking_ids.empty:
            errors.append(
                "content IDs cross split boundaries "
                f"(first: {str(leaking_ids.index[0])!r})"
            )

    real_with_fake_family = frame[
        (frame["label"] == 0) & (frame["generator_family"] != "real")
    ]
    if not real_with_fake_family.empty:
        errors.append("authentic records must use generator_family='real'")
    fake_with_real_family = frame[
        (frame["label"] == 1) & (frame["generator_family"] == "real")
    ]
    if not fake_with_real_family.empty:
        errors.append("generated/manipulated records cannot use generator_family='real'")

    if forbid_demo_sources:
        prohibited = prohibited_demo_mask(frame)
        if prohibited.any():
            reason = prohibited_demo_reason(frame.loc[prohibited].iloc[0])
            errors.append(f"prohibited demonstration source found: {reason}")

    if check_paths:
        missing_paths: list[str] = []
        for raw_path in frame["path"]:
            path = Path(raw_path)
            if not path.is_absolute():
                path = project_root / path
            if not path.is_file():
                missing_paths.append(str(path))
                if len(missing_paths) == 3:
                    break
        if missing_paths:
            errors.append(f"local image files are missing: {missing_paths}")

    if errors:
        raise ManifestValidationError("; ".join(errors))


def write_manifest(
    records: pd.DataFrame | Iterable[dict[str, object]],
    output_path: str | Path,
    *,
    check_paths: bool = False,
    forbid_demo_sources: bool = True,
) -> pd.DataFrame:
    """Normalise, validate, and atomically write a CSV manifest."""

    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    frame = normalise_manifest(frame)
    validate_manifest(
        frame,
        check_paths=check_paths,
        forbid_demo_sources=forbid_demo_sources,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)
    return frame
