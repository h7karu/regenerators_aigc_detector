"""Predict AIGC probabilities for every supported image in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from augmentations import build_eval_transform
from model import build_model, load_checkpoint
from utils import atomic_write_json, load_config, select_device


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class ImageDirectoryDataset(Dataset):
    def __init__(
        self, directory: Path, image_size: int, limit: int | None = None
    ) -> None:
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        self.paths = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not self.paths:
            raise ValueError(f"No supported images found in {directory}")
        if limit is not None:
            self.paths = self.paths[:limit]
        self.transform = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.paths[index]
        try:
            with Image.open(path) as source:
                image = np.asarray(
                    ImageOps.exif_transpose(source).convert("RGB"), dtype=np.uint8
                )
        except Exception as error:
            raise RuntimeError(f"Unable to decode image: {path}") from error
        return {"image": self.transform(image=image)["image"], "path": str(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/rgb_baseline.yaml")
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit files for smoke tests; omit for normal inference.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = select_device(args.device)
    model = build_model(config, pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    dataset = ImageDirectoryDataset(
        Path(args.input_dir), int(config["data"]["image_size"]), limit=args.limit
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    predictions: list[dict[str, object]] = []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["image"].to(device, non_blocking=True))
            probabilities = logits.sigmoid().cpu().tolist()
            predictions.extend(
                {"image_path": path, "pred": float(probability)}
                for path, probability in zip(batch["path"], probabilities)
            )
    atomic_write_json(predictions, args.output)
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
