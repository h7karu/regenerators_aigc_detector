"""Download a small WildFake subset from ModelScope into the CIFAKE folder layout.

WildFake is ~1.3 TB of monolithic ZIP archives (6-54 GB each), which is far too
much to pull for a hackathon prototype. Instead of downloading any archive, we
read each ZIP's central directory over HTTP range requests and fetch only the
few hundred image members we actually sample -- a few MB of traffic total.
See `aigc_detector/data/remote_zip.py` for the mechanics.

LEAKAGE GUARD: the organisers reserved a WildFake subset as a demo-only
benchmark -- COCO val2017 (non-AIGC) and DALL-E 3 "Advanced" (AIGC). Those must
never be used for training, so the training sources below deliberately exclude
`coco/coco2017/val2017` and `DALLE/Advanced`. Use
`download_wildfake_benchmark.py` to fetch the benchmark separately.

Usage:
    python -m aigc_detector.data.download_wildfake --per-class 300
"""
from __future__ import annotations

import argparse
import urllib.parse
from pathlib import Path

from aigc_detector.data.prepare import (
    normalize_image_bytes,
    sample_deterministic,
    split_counts,
    summarize,
    target_dir,
    write_bytes,
)
from aigc_detector.data.remote_zip import open_remote_zip

MODELSCOPE_DATASET = "hy2628982280/WildFake"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# (archive path, member-prefix filter, short tag used in output filenames).
# `None` prefix means "any image member in the archive".
REAL_SOURCES = {
    # COCO restricted to train2017: val2017 is the reserved benchmark split.
    "coco": ("Images/Real/coco.zip", "coco/coco2017/train2017/", "coco"),
    "imagenet": ("Images/Real/imagenet.zip", None, "imagenet"),
    "celebahq": ("Images/Real/celebahq.zip", None, "celebahq"),
    "afhq": ("Images/Real/afhq.zip", None, "afhq"),
    "church": ("Images/Real/church.zip", None, "church"),
    "ffhq": ("Images/Real/ffhq.zip", None, "ffhq"),
}

# Multiple generator families, so the model sees more than one kind of artifact.
FAKE_SOURCES = {
    "ddim": ("Images/Diffusion_based/DDIM.zip", None, "ddim"),
    "ddpm": ("Images/Diffusion_based/DDPM.zip", None, "ddpm"),
    "vqdm": ("Images/Diffusion_based/VQDM.zip", None, "vqdm"),
    "adm": ("Images/Diffusion_based/ADM.zip", None, "adm"),
    "imagen": ("Images/Diffusion_based/Imagen.zip", None, "imagen"),
    # DALL-E restricted to the Typical split: Advanced (DALL-E 3) is reserved.
    "dalle2": ("Images/Diffusion_based/DALLE.zip", "DALLE/Typical/", "dalle2"),
}

DEFAULT_REAL = ["coco", "imagenet", "celebahq"]
DEFAULT_FAKE = ["ddim", "ddpm", "vqdm"]


def archive_url(path: str) -> str:
    return (
        f"https://modelscope.cn/api/v1/datasets/{MODELSCOPE_DATASET}/repo?Revision=master&FilePath="
        + urllib.parse.quote(path, safe="")
    )


def collect_from_archive(
    archive_path: str,
    prefix: str | None,
    tag: str,
    n_wanted: int,
    seed: int,
    normalize: bool = True,
) -> list[tuple[str, bytes]]:
    """Sample `n_wanted` image members from a remote archive and return their bytes."""
    print(f"  opening {archive_path} (reading central directory over HTTP range)...")
    zf = open_remote_zip(archive_url(archive_path))
    members = [
        n
        for n in zf.namelist()
        if n.lower().endswith(IMAGE_EXTS) and (prefix is None or n.startswith(prefix))
    ]
    if not members:
        raise RuntimeError(f"No image members matched prefix {prefix!r} in {archive_path}")
    print(f"    {len(members)} candidate members; sampling {min(n_wanted, len(members))}")

    picked = sample_deterministic(members, n_wanted, seed)
    out: list[tuple[str, bytes]] = []
    for i, name in enumerate(picked, 1):
        payload = zf.read(name)
        if normalize:
            payload, ext = normalize_image_bytes(payload)
        else:
            ext = Path(name).suffix.lower()
        out.append((f"{tag}_{i:05d}{ext}", payload))
        if i % 100 == 0:
            print(f"    fetched {i}/{len(picked)}")
    return out


def gather(
    sources: dict,
    keys: list[str],
    per_class: int,
    seed: int,
    test_fraction: float,
    normalize: bool = True,
) -> tuple[list[tuple[str, bytes]], list[tuple[str, bytes]]]:
    """Pull roughly `per_class` images spread evenly across the chosen sources.

    Splits each source into train/test *before* merging across sources. Doing
    the split on the merged list instead (as an earlier version of this script
    did) puts whole sources entirely on one side: sources are fetched one
    archive at a time and appended as contiguous blocks, so a flat train/test
    cut just slices off the last block(s) wholesale (e.g. the WildFake test
    split ended up 100% celebahq for REAL and 100% VQDM for FAKE). Splitting
    per-source keeps every source represented in both splits.
    """
    train_items: list[tuple[str, bytes]] = []
    test_items: list[tuple[str, bytes]] = []
    n_each = max(1, per_class // len(keys))
    n_collected = 0
    for i, key in enumerate(keys):
        archive_path, prefix, tag = sources[key]
        # Give the last source any remainder so totals land on `per_class`.
        want = per_class - n_collected if i == len(keys) - 1 else n_each
        if want <= 0:
            break
        source_items = collect_from_archive(archive_path, prefix, tag, want, seed, normalize)
        n_collected += len(source_items)

        n_train, _ = split_counts(len(source_items), test_fraction)
        train_items.extend(source_items[:n_train])
        test_items.extend(source_items[n_train:])
    return train_items, test_items


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--per-class",
        type=int,
        default=300,
        help="Total images per class (REAL/FAKE) across train+test. Keep small: this is a subset.",
    )
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--dataset-name", default="wildfake")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--real-sources", nargs="+", default=DEFAULT_REAL, choices=sorted(REAL_SOURCES))
    p.add_argument("--fake-sources", nargs="+", default=DEFAULT_FAKE, choices=sorted(FAKE_SOURCES))
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Keep original bytes instead of re-encoding. Leaves the format/size label leak intact.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Fetching a WildFake subset: {args.per_class} REAL + {args.per_class} FAKE\n"
        f"  real sources: {', '.join(args.real_sources)}\n"
        f"  fake sources: {', '.join(args.fake_sources)}\n"
        "  (streaming individual ZIP members; the archives themselves are NOT downloaded)\n"
        f"  normalization: {'OFF (format/size leak left intact)' if args.no_normalize else 'ON'}\n"
    )

    for label, sources, keys in (
        (0, REAL_SOURCES, args.real_sources),
        (1, FAKE_SOURCES, args.fake_sources),
    ):
        name = "REAL" if label == 0 else "FAKE"
        print(f"[{name}]")
        train_items, test_items = gather(
            sources, keys, args.per_class, args.seed, args.test_fraction, not args.no_normalize
        )

        for split, split_items in (("train", train_items), ("test", test_items)):
            for filename, payload in split_items:
                write_bytes(target_dir(args.data_root, args.dataset_name, split, label), filename, payload)
        print(f"  wrote {len(train_items)} train + {len(test_items)} test {name} images\n")

    summarize(args.data_root, args.dataset_name)
    print(
        "\nNOTE: this subset excludes COCO val2017 and DALL-E 3 'Advanced' by design -- "
        "those are the organisers' reserved benchmark and must not be trained on."
    )


if __name__ == "__main__":
    main()
