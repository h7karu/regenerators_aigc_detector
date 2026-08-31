"""Split a materialized SID validation pool into selection and holdout sets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from manifest_schema import normalise_manifest, validate_manifest, write_manifest  # noqa: E402
from utils import resolve_project_path  # noqa: E402


def stratified_split(
    frame: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split every SID subtype independently and deterministically."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    frame = normalise_manifest(frame)
    validate_manifest(frame)
    if set(frame["dataset"]) != {"sid"}:
        raise ValueError("The input manifest must contain only dataset='sid'.")

    rng = np.random.default_rng(seed)
    validation_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("original_label", sort=True):
        indices = rng.permutation(group.index.to_numpy())
        validation_count = int(round(len(indices) * validation_fraction))
        if validation_count <= 0 or validation_count >= len(indices):
            raise ValueError("Every SID subtype needs at least two examples.")
        validation_parts.append(frame.loc[indices[:validation_count]].copy())
        holdout_parts.append(frame.loc[indices[validation_count:]].copy())

    validation = pd.concat(validation_parts, ignore_index=True)
    holdout = pd.concat(holdout_parts, ignore_index=True)
    validation = validation.iloc[rng.permutation(len(validation))].reset_index(drop=True)
    holdout = holdout.iloc[rng.permutation(len(holdout))].reset_index(drop=True)
    validation["split"] = "val"
    holdout["split"] = "test"

    combined = normalise_manifest(pd.concat([validation, holdout], ignore_index=True))
    validate_manifest(combined)
    return normalise_manifest(validation), normalise_manifest(holdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/manifests/sid_val.csv")
    parser.add_argument(
        "--validation-output",
        default="data/manifests/sid_model_val.csv",
    )
    parser.add_argument(
        "--holdout-output",
        default="data/manifests/sid_test.csv",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = normalise_manifest(pd.read_csv(resolve_project_path(args.input)))
    validation, holdout = stratified_split(
        source,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    validation_path = resolve_project_path(args.validation_output)
    holdout_path = resolve_project_path(args.holdout_output)
    validation = write_manifest(validation, validation_path, check_paths=True)
    holdout = write_manifest(holdout, holdout_path, check_paths=True)
    print(f"Wrote {len(validation):,} model-selection rows to {validation_path}")
    print(f"Wrote {len(holdout):,} holdout rows to {holdout_path}")
    print(
        pd.concat([validation, holdout])
        .groupby(["split", "original_label", "generator"])
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
