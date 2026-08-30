"""Robustness transforms matching the hackathon brief's evaluation table.

Each transform takes and returns a PIL Image in the same size as the input,
so a transformed test set can be scored with the same pipeline as clean data.
`ROBUSTNESS_TRANSFORMS` maps a human-readable variant name to a callable and
is what `evaluate.py` iterates over to build the robustness summary table.
"""
from __future__ import annotations

import io
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def identity(image: Image.Image) -> Image.Image:
    return image


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_roundtrip(image: Image.Image, scale: float) -> Image.Image:
    """Downscale by `scale` then upscale back, simulating thumbnailing."""
    w, h = image.size
    small = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    """`sigma` is in normalized [0, 1] pixel units (e.g. 0.05 -> ~12.75/255)."""
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    noisy = arr + np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def color_jitter(image: Image.Image, factor: float) -> Image.Image:
    """Apply brightness/contrast/saturation all scaled by `factor` (e.g. 1.2 = +20%)."""
    out = image.convert("RGB")
    out = ImageEnhance.Brightness(out).enhance(factor)
    out = ImageEnhance.Contrast(out).enhance(factor)
    out = ImageEnhance.Color(out).enhance(factor)
    return out


def center_crop(image: Image.Image, fraction: float) -> Image.Image:
    """Crop the center `fraction` of width/height, then resize back to the original size."""
    w, h = image.size
    cw, ch = round(w * fraction), round(h * fraction)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = image.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BICUBIC)


# Name -> transform. Parameters come directly from the brief's robustness table.
ROBUSTNESS_TRANSFORMS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "clean": identity,
    "jpeg_q90": lambda img: jpeg_compress(img, 90),
    "jpeg_q70": lambda img: jpeg_compress(img, 70),
    "jpeg_q50": lambda img: jpeg_compress(img, 50),
    "jpeg_q30": lambda img: jpeg_compress(img, 30),
    "blur_sigma0.5": lambda img: gaussian_blur(img, 0.5),
    "blur_sigma1.0": lambda img: gaussian_blur(img, 1.0),
    "blur_sigma2.0": lambda img: gaussian_blur(img, 2.0),
    "resize_0.5x": lambda img: resize_roundtrip(img, 0.5),
    "resize_0.25x": lambda img: resize_roundtrip(img, 0.25),
    "noise_sigma0.02": lambda img: gaussian_noise(img, 0.02),
    "noise_sigma0.05": lambda img: gaussian_noise(img, 0.05),
    "noise_sigma0.10": lambda img: gaussian_noise(img, 0.10),
    "color_jitter+20%": lambda img: color_jitter(img, 1.2),
    "color_jitter-20%": lambda img: color_jitter(img, 0.8),
    "center_crop_80%": lambda img: center_crop(img, 0.8),
}


# ---------------------------------------------------------------------------
# Train-time augmentation
#
# The eval grid above uses the brief's exact fixed parameters. For *training* we
# deliberately sample severities from continuous ranges instead, so the model
# learns to tolerate degradation in general rather than memorising the specific
# settings it will be scored on. Ranges extend slightly past the eval values so
# the eval points sit inside the training distribution rather than at its edge.
# ---------------------------------------------------------------------------

import random  # noqa: E402  (kept next to the augmentation code it belongs to)


def _random_ops(rng: random.Random) -> list[Callable[[Image.Image], Image.Image]]:
    return [
        lambda img: jpeg_compress(img, rng.randint(25, 95)),
        lambda img: gaussian_blur(img, rng.uniform(0.3, 2.2)),
        lambda img: resize_roundtrip(img, rng.uniform(0.2, 0.8)),
        lambda img: gaussian_noise(img, rng.uniform(0.01, 0.12)),
        lambda img: color_jitter(img, rng.uniform(0.75, 1.25)),
        lambda img: center_crop(img, rng.uniform(0.7, 0.95)),
    ]


def random_augment(
    image: Image.Image,
    rng: random.Random,
    max_ops: int = 2,
    p_clean: float = 0.0,
) -> Image.Image:
    """Apply 1-`max_ops` randomly chosen transforms at random severity.

    Stacking up to two operations mimics real redistribution chains (e.g. a
    screenshot that is then re-compressed by a messaging app).
    """
    if p_clean > 0 and rng.random() < p_clean:
        return image
    ops = _random_ops(rng)
    n = rng.randint(1, max(1, max_ops))
    out = image
    for op in rng.sample(ops, n):
        out = op(out)
    return out


def build_training_views(
    image: Image.Image,
    rng: random.Random,
    n_augmented: int = 1,
    keep_clean: bool = True,
    max_ops: int = 2,
) -> list[Image.Image]:
    """Return the training views for one source image.

    Keeping the clean view alongside augmented copies stops the model from
    trading away clean accuracy for robustness -- the brief scores both.
    """
    views = [image] if keep_clean else []
    views.extend(random_augment(image, rng, max_ops=max_ops) for _ in range(n_augmented))
    return views
