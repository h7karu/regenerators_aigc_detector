"""Download a small SID_Set subset from Hugging Face into the CIFAKE folder layout.

SID_Set is ~140 GB across 249 parquet shards, so we never fetch a whole shard.
Parquet is columnar and stores per-row-group offsets in its footer, so with HTTP
range requests we can:

  * read just the footer to learn the row-group layout,
  * read only the `image` and `label` columns -- skipping the large `mask`
    column entirely, which is roughly half the payload,
  * stop as soon as we have enough images per class.

Fetching a few hundred images costs a few hundred MB of transfer instead of
140 GB. See `aigc_detector/data/remote_zip.py` for the range-request file object.

Labels in SID_Set are 0 = real, 1 = fully synthetic, 2 = tampered. We map
0 -> REAL and 1 -> FAKE. Tampered images are only *partially* manipulated, which
is a different task from the brief's image-level "is this AI-generated?", so
they are excluded unless `--include-tampered` is passed.

Usage:
    python -m aigc_detector.data.download_sid_set --per-class 150
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

from aigc_detector.data.prepare import (
    normalize_image_bytes,
    split_counts,
    summarize,
    target_dir,
    write_bytes,
)
from aigc_detector.data.remote_zip import HttpRangeFile

REPO = "saberzl/SID_Set"
N_TRAIN_SHARDS = 249

SID_REAL, SID_SYNTHETIC, SID_TAMPERED = 0, 1, 2


def shard_url(index: int) -> str:
    return (
        f"https://huggingface.co/datasets/{REPO}/resolve/main/"
        f"data/train-{index:05d}-of-{N_TRAIN_SHARDS:05d}.parquet"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--per-class",
        type=int,
        default=150,
        help="Images per class (REAL/FAKE) across train+test. Keep small: images are ~0.5 MB each.",
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--dataset-name", default="sid_set")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument(
        "--include-tampered",
        action="store_true",
        help="Also map SID_Set's tampered class (label 2) to FAKE.",
    )
    p.add_argument(
        "--max-shards",
        type=int,
        default=8,
        help="Safety cap on how many parquet shards to touch.",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep original bytes instead of re-encoding. Leaves the format/size label leak intact.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import pyarrow.parquet as pq

    fake_labels = {SID_SYNTHETIC} | ({SID_TAMPERED} if args.include_tampered else set())
    print(
        f"Fetching a SID_Set subset: {args.per_class} REAL + {args.per_class} FAKE\n"
        f"  tampered images: {'included as FAKE' if args.include_tampered else 'excluded'}\n"
        "  (reading only the image+label columns of selected row groups)\n"
        f"  normalization: {'OFF (format/size leak left intact)' if args.no_normalize else 'ON'}\n"
    )

    collected: dict[int, list[bytes]] = {0: [], 1: []}
    shard = 0
    while shard < args.max_shards and any(
        len(v) < args.per_class for v in collected.values()
    ):
        url = shard_url(shard)
        print(f"[shard {shard}] {url.rsplit('/', 1)[-1]}")
        try:
            pf = pq.ParquetFile(io.BufferedReader(HttpRangeFile(url), buffer_size=1 << 20))
        except Exception as exc:  # noqa: BLE001 - keep going if one shard is unavailable
            print(f"  could not open shard ({exc}); skipping")
            shard += 1
            continue

        for rg in range(pf.metadata.num_row_groups):
            if all(len(v) >= args.per_class for v in collected.values()):
                break
            # Read labels first (cheap) so we can skip row groups we don't need.
            labels = pf.read_row_group(rg, columns=["label"]).column("label").to_pylist()
            wanted = {
                i: (0 if lab == SID_REAL else 1)
                for i, lab in enumerate(labels)
                if (lab == SID_REAL and len(collected[0]) < args.per_class)
                or (lab in fake_labels and len(collected[1]) < args.per_class)
            }
            if not wanted:
                continue

            table = pf.read_row_group(rg, columns=["image"]).column("image").to_pylist()
            for i, target in wanted.items():
                if len(collected[target]) < args.per_class:
                    collected[target].append(table[i]["bytes"])
            print(
                f"  row group {rg}: REAL {len(collected[0])}/{args.per_class}, "
                f"FAKE {len(collected[1])}/{args.per_class}"
            )
        shard += 1

    if not any(collected.values()):
        raise SystemExit("No images collected -- check network access to huggingface.co")

    for label, payloads in collected.items():
        name = "REAL" if label == 0 else "FAKE"
        n_train, _ = split_counts(len(payloads), args.test_fraction)
        for i, payload in enumerate(payloads):
            split = "train" if i < n_train else "test"
            if args.no_normalize:
                ext = ".png" if payload[:4] == b"\x89PNG" else ".jpg"
            else:
                payload, ext = normalize_image_bytes(payload)
            write_bytes(
                target_dir(args.data_root, args.dataset_name, split, label),
                f"sid_{name.lower()}_{i:05d}{ext}",
                payload,
            )
        print(f"wrote {len(payloads)} {name} images")

    summarize(args.data_root, args.dataset_name)


if __name__ == "__main__":
    main()
