"""Forensic feature branch: frequency-domain + noise-residual descriptors.

Motivation. CLIP was trained to align images with captions, so its embedding is
tuned for *semantic* content and tends to discard the very high-frequency detail
that betrays a generator. Classical image forensics goes the other way: it
throws away content and keeps the sensor/texture statistics. Generative models
built on up-convolution (GANs, most diffusion decoders) leave characteristic
periodic artifacts in the frequency spectrum, and their noise residuals differ
statistically from real camera noise.

This module produces a compact, hand-engineered descriptor from three families:

  1. Radially-averaged power spectrum (RAPS) of the FFT -- the standard tool for
     exposing up-sampling grid artifacts (cf. Durall et al., Zhang et al.).
  2. High-pass "noise residual" statistics using fixed SRM-style kernels
     borrowed from steganalysis -- captures how the fine texture is distributed.
  3. Block-DCT statistics -- sensitive to both generator artifacts and the
     compression history of the image.

These are fixed (untrained) transforms, so the branch adds no parameters to the
model at all -- it just gives the trained head a view of the image that CLIP
does not provide.

Note on resolution: all features are computed on a canonical grayscale resize so
that spectra are comparable across differently-sized inputs. This costs some
absolute high-frequency detail but makes the descriptor well-defined for a
dataset that mixes 32x32 (CIFAKE) with megapixel images (SID_Set/WildFake).
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fft import dctn

CANONICAL_SIZE = 256
N_RADIAL_BINS = 48

# Fixed high-pass kernels in the spirit of SRM (spatial rich model) residuals.
# Each suppresses image content and leaves noise/texture behind.
SRM_KERNELS: list[np.ndarray] = [
    # first-order horizontal difference
    np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32),
    # second-order horizontal
    np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32),
    # 3x3 Laplacian-like ("square" residual)
    np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) / 4.0,
    # 5x5 KV kernel, a classic steganalysis residual filter
    np.array(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=np.float32,
    )
    / 12.0,
]


def _to_canonical_gray(image: Image.Image) -> np.ndarray:
    """Grayscale float array in [0,1] at a fixed size, so spectra are comparable."""
    gray = image.convert("L").resize((CANONICAL_SIZE, CANONICAL_SIZE), Image.BICUBIC)
    return np.asarray(gray, dtype=np.float32) / 255.0


def _convolve2d_valid(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Small 2-D correlation via stride tricks (avoids a scipy.signal dependency)."""
    kh, kw = kernel.shape
    view = np.lib.stride_tricks.sliding_window_view(arr, (kh, kw))
    return np.einsum("ijkl,kl->ij", view, kernel)


def _moments(x: np.ndarray) -> list[float]:
    """Scale/shape descriptors of a residual map."""
    x = x.ravel()
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-12:
        return [mean, 0.0, 0.0, 0.0, 0.0, 0.0]
    z = (x - mean) / std
    return [
        mean,
        std,
        float(np.mean(z**3)),  # skewness
        float(np.mean(z**4)),  # kurtosis
        float(np.mean(np.abs(x))),
        float(np.percentile(np.abs(x), 99)),
    ]


def radial_power_spectrum(gray: np.ndarray, n_bins: int = N_RADIAL_BINS) -> np.ndarray:
    """Azimuthally-averaged log power spectrum: a 1-D profile from DC to Nyquist.

    Up-convolution artifacts show up as bumps/plateaus at the high-frequency end
    of this curve, which is what makes it discriminative for generated images.
    """
    windowed = gray - gray.mean()
    # Hann window in both axes to suppress edge-induced spectral leakage.
    win = np.hanning(windowed.shape[0])[:, None] * np.hanning(windowed.shape[1])[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2(windowed * win))
    power = np.log1p(np.abs(spectrum) ** 2)

    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    max_r = float(radius[cy, cx:].max())

    bins = np.linspace(0, max_r, n_bins + 1)
    idx = np.clip(np.digitize(radius.ravel(), bins) - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=power.ravel(), minlength=n_bins)
    counts = np.bincount(idx, minlength=n_bins)
    profile = sums / np.maximum(counts, 1)

    # Normalize out overall brightness/contrast: the *shape* carries the signal.
    return profile - profile.mean()


def residual_features(gray: np.ndarray) -> np.ndarray:
    """Moments of several fixed high-pass residuals, plus their cross-correlation."""
    feats: list[float] = []
    residuals = []
    for kernel in SRM_KERNELS:
        res = _convolve2d_valid(gray, kernel)
        residuals.append(res)
        feats.extend(_moments(res))

    # How similar the different residual maps are to each other -- real sensor
    # noise and synthesis artifacts differ in how correlated these views are.
    for i in range(len(residuals)):
        for j in range(i + 1, len(residuals)):
            a, b = residuals[i], residuals[j]
            n = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
            av = a[: n[0], : n[1]].ravel()
            bv = b[: n[0], : n[1]].ravel()
            denom = np.linalg.norm(av) * np.linalg.norm(bv)
            feats.append(float(av @ bv / denom) if denom > 1e-12 else 0.0)
    return np.asarray(feats, dtype=np.float32)


def dct_features(gray: np.ndarray, block: int = 8) -> np.ndarray:
    """Mean log-magnitude per position in an 8x8 block DCT, plus a high/low ratio.

    Captures periodic compression-style structure and generator fingerprints.
    """
    h = gray.shape[0] // block * block
    w = gray.shape[1] // block * block
    cropped = gray[:h, :w]
    blocks = cropped.reshape(h // block, block, w // block, block).transpose(0, 2, 1, 3)
    coeffs = dctn(blocks, axes=(2, 3), norm="ortho")
    mag = np.log1p(np.abs(coeffs)).mean(axis=(0, 1))  # (block, block)

    flat = mag.ravel()
    # Ratio of high-frequency to low-frequency energy (DC excluded).
    freq_idx = np.add.outer(np.arange(block), np.arange(block))
    low = mag[(freq_idx <= 2) & (freq_idx > 0)].mean()
    high = mag[freq_idx >= block].mean()
    ratio = float(high / low) if low > 1e-12 else 0.0
    return np.concatenate([flat, [ratio]]).astype(np.float32)


def extract_forensic_features(image: Image.Image) -> np.ndarray:
    """Full forensic descriptor for one image."""
    gray = _to_canonical_gray(image)
    return np.concatenate(
        [
            radial_power_spectrum(gray),
            residual_features(gray),
            dct_features(gray),
        ]
    ).astype(np.float32)


def extract_forensic_features_batch(images: list[Image.Image]) -> np.ndarray:
    return np.stack([extract_forensic_features(img) for img in images])


FORENSIC_FEATURE_DIM = N_RADIAL_BINS + len(SRM_KERNELS) * 6 + (len(SRM_KERNELS) * (len(SRM_KERNELS) - 1)) // 2 + 65
