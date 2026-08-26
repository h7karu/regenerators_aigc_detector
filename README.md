# regenerators_aigc_detector
This repository contains team regenerator's Artificial Intelligence Generated Content (AIGC) detector for TikTok TechJam 2026.

## Project overview

The task is to tell AI-generated images from authentic ones, and to keep working
after the image has been compressed, blurred, cropped, resized or filtered on its
way through a social platform. Clean-data accuracy is the easy half; the score
lives in what survives post-processing.

Our detector is **dual-branch**. Each branch sees something the other misses:

```
image ──┬──> frozen CLIP ViT-B/32 ─────> 512-d semantic embedding ──┐
        │                                                            ├─> scale ─> head ─> P(AIGC)
        └──> fixed forensic transforms ─> 143-d artifact vector ─────┘
             (FFT radial spectrum, SRM
              noise residuals, block DCT)
```

- **Semantic branch (CLIP).** A frozen, large-scale-pretrained vision encoder.
  Its features are stable under exactly the degradations we are scored on, but
  CLIP was trained to match images to captions, so it is tuned for *content* and
  tends to smooth away the fine detail that betrays a generator.
- **Forensic branch.** Fixed, untrained signal-processing descriptors that throw
  away content and keep the artifact statistics: the radially-averaged FFT power
  spectrum (exposes the periodic grid left by up-convolution layers), SRM-style
  high-pass noise residuals borrowed from steganalysis, and block-DCT statistics.

They fail in different places, which is the point. Heavy blur and downscaling
destroy the high frequencies the forensic branch depends on, while CLIP still
recognises the image; conversely CLIP is weakest exactly where generator
fingerprints are most obvious. Fusing them means a transform that defeats one
branch usually leaves the other standing.

Both branches are **frozen** — only a small scikit-learn head is trained. So the
whole model is ~150M parameters (well under the 2B cap), trains in minutes on a
laptop CPU, and needs no GPU.

**Robustness is trained in, not just measured.** Every source image contributes a
clean view plus randomly degraded copies (JPEG, blur, resize, noise, colour
jitter, crop, at random severity, stacked up to 2 deep). Training severities are
sampled from continuous ranges wider than the evaluation grid, so the model
learns to tolerate degradation in general rather than memorising the exact
settings it will be tested on.

### Repo layout

| Path | What it is |
|---|---|
| [aigc_detector/train.py](aigc_detector/train.py) | Train the detector |
| [aigc_detector/infer.py](aigc_detector/infer.py) | Score an image directory → JSON |
| [aigc_detector/evaluate.py](aigc_detector/evaluate.py) | Robustness table + error analysis |
| [aigc_detector/models/dual_branch.py](aigc_detector/models/dual_branch.py) | The fusion model |
| [aigc_detector/features/forensic.py](aigc_detector/features/forensic.py) | Frequency / noise-residual branch |
| [aigc_detector/data/transforms.py](aigc_detector/data/transforms.py) | Robustness transforms + train-time augmentation |
| [aigc_detector/data/](aigc_detector/data/) | Dataset loaders and subset downloaders |
| [scripts/](scripts/) | One download script per dataset |

## Setup and installation instructions

### 1. Get the repo and the virtual environment

```
git pull origin main
```

Add the `.env` file to the repo root. It holds the Kaggle API credentials used by
the CIFAKE download and **must not be committed** (it is already gitignored).

Set up the virtual environment. On WSL/Ubuntu you may first need
`sudo apt install python3-venv`:

```
python3 -m venv venv
source venv/bin/activate
```

Your prompt should now be prefixed with `(venv)`, which means you are inside the
virtual environment.

### 2. Install the packages

```
pip install -r requirements.txt
```

`requirements.txt` gets updated when new packages are needed — just run this
again after a `git pull`.

### 3. Download the datasets

Everything under `data/` is gitignored, so **the datasets are never pushed to
GitHub** — each person downloads them locally with these scripts. Run them from
the repo root.

The two large datasets are only ever fetched as **small subsets**. SID_Set is
~140 GB and WildFake is ~1.3 TB in full, so the scripts never download the whole
thing: they use HTTP range requests to pull just the individual images they need
(a few hundred MB for SID_Set, a few MB for WildFake).

```bash
# CIFAKE — the full dataset (~100k images), via the Kaggle API
chmod +x scripts/*.sh
./scripts/download_cifake.sh

# SID_Set subset (default 150 images per class)
./scripts/download_sid_set.sh
./scripts/download_sid_set.sh --per-class 300     # or ask for more

# WildFake subset (default 300 images per class, spread across generator families)
./scripts/download_wildfake.sh
./scripts/download_wildfake.sh --per-class 500

# The organisers' reserved demo benchmark — EVALUATION ONLY, never train on this
./scripts/download_wildfake_benchmark.sh
```

Each script writes the same folder layout, so everything downstream treats them
identically:

```
data/<dataset>/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

Start small. Every script takes `--per-class`, and the defaults are deliberately
modest so a laptop does not fill up. Check what you have at any time with
`du -sh data/*`.

> **Reserved benchmark.** `download_wildfake_benchmark.sh` fetches COCO val2017
> (non-AIGC) and DALL·E 3 "Advanced" (AIGC), which the organisers set aside to
> demo performance. It does not count toward the final score and **must not be
> trained on**. It lands in `data/wildfake_benchmark/eval/` — an `eval/` split,
> not `train/`, so a training run cannot pick it up by accident. The training
> downloaders exclude those same images at the source.

## Steps to reproduce the results

### Train

```bash
# Quick baseline on CIFAKE
python -m aigc_detector.train --data-dir data/cifake/train --max-per-class 2000

# Mix datasets — this is what generalises across generator families
python -m aigc_detector.train \
    --data-dir data/cifake/train \
    --data-dir data/wildfake/train \
    --data-dir data/sid_set/train \
    --max-per-class 1000 \
    --output models/dual_branch.joblib
```

Useful flags:

| Flag | Default | Why you'd change it |
|---|---|---|
| `--max-per-class N` | all | Cap source images per class. Start small. |
| `--augment-copies N` | `2` | Degraded copies per image. `0` disables augmentation. |
| `--branches` | `clip forensic` | Use one branch alone for ablations. |
| `--head` | `logreg` | `mlp` for a non-linear fusion head. |

Training cost scales as `images × (1 + augment-copies)`, so `--max-per-class
1000 --augment-copies 2` means 6000 feature extractions.

### Predict on a directory (the required deliverable)

```bash
python -m aigc_detector.infer \
    --input-dir path/to/images \
    --checkpoint models/dual_branch.joblib \
    --output predictions.json
```

Writes the format the brief asks for — `pred` is the confidence the image is
AI-generated:

```json
[
  {"image_path": "path/to/images/a.jpg", "pred": 0.9412},
  {"image_path": "path/to/images/b.jpg", "pred": 0.0317}
]
```

### Evaluate robustness

```bash
# Held-out CIFAKE test split
python -m aigc_detector.evaluate \
    --data-dir data/cifake/test \
    --checkpoint models/dual_branch.joblib

# The organisers' reserved benchmark
python -m aigc_detector.evaluate \
    --data-dir data/wildfake_benchmark/eval \
    --checkpoint models/dual_branch.joblib \
    --output-dir reports/benchmark
```

This scores the test set clean and under every transform in the brief's table,
and writes to `reports/`:

- `robustness_summary.csv` / `.md` — accuracy, precision, recall, F1 and AUROC
  per transform, plus the drop against clean.
- `error_analysis.json` — the most confident false positives and false negatives.

**On the operating threshold.** Accuracy at a 0.5 cutoff is a weak summary for a
moderation system: wrongly flagging a real user's photo is the more expensive
mistake, so the two error types should not be traded one-for-one. The evaluator
therefore also picks the threshold that holds the false-positive rate at or below
`--target-fpr` (default 5%) on clean data, and reports every transform at that
same fixed threshold. `recall@fixed_thr` is the number to watch: it says how much
AIGC we still catch at a false-positive rate a platform could actually live with.

### Reproduce the ablation

The fusion should be justified, not assumed — train each branch alone and compare:

```bash
for B in "clip" "forensic" "clip forensic"; do
  python -m aigc_detector.train --branches $B --max-per-class 1000 \
      --output "models/ablation_${B// /_}.joblib"
  python -m aigc_detector.evaluate --checkpoint "models/ablation_${B// /_}.joblib" \
      --output-dir "reports/ablation_${B// /_}"
done
```

## Limitations

- **CIFAKE is 32×32.** CLIP expects 224×224, so CIFAKE images are upscaled ~7×
  before the encoder sees them, and there is little genuine high-frequency detail
  for the forensic branch to read. CIFAKE is therefore a fast iteration loop, not
  a realistic proxy for social-media imagery — trust the SID_Set and WildFake
  numbers more.
- **Dataset shortcuts.** In both SID_Set and WildFake the two classes arrive in
  systematically different containers (real images as JPEGs at assorted sizes,
  synthetic ones as 1024×1024 PNGs). A model can score near-perfectly by keying
  on "PNG and square" while learning nothing about generation — and that shortcut
  vanishes the moment it meets a real PNG. The downloaders re-encode every image
  through the same encoder at the same size cap to remove it (`--no-normalize`
  keeps the original bytes if you want to measure the difference). One residual
  cue survives: the synthetic images are square and many real ones are not, which
  we do not correct because cropping would destroy more signal than it removes.
- **Frozen backbone.** Only the head is trained, so accuracy is capped by what is
  linearly separable in the fused feature space. Fine-tuning the last few CLIP
  blocks is the obvious next step, and still fits the parameter budget.
- **Generator coverage.** The subsets cover a handful of generator families. The
  reserved benchmark is DALL·E 3, which nothing in training has seen — that gap
  is the honest test of generalisation, and we expect it to be the weakest
  number.
- **The forensic branch has a known blind spot.** It reads a fixed 256×256
  grayscale resize, so aggressive downscaling (0.25×) or heavy blur removes the
  evidence it depends on. This is visible in the robustness table and is exactly
  why the fusion exists.

### What we'd do with more time

1. Fine-tune the last CLIP blocks instead of freezing everything.
2. Train on many more generator families and measure leave-one-generator-out
   generalisation, which is the metric that predicts real deployment.
3. Add a calibration step (temperature scaling / reliability diagrams) so `pred`
   can be read as a probability rather than just ranked.
4. Patch-level scoring with aggregation, so high-resolution images are not
   bottlenecked through a single 224×224 resize.

## Contribution
Team member contributions (if applicable, i.e. team participants, non-solo participants)
