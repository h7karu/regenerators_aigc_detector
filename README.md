# Regenerators AIGC Detector

Team Regenerators' robust AI-generated image detector for TikTok TechJam 2026.

## Quick start

The reproducible local environment uses Python 3.12.14 and a hash-locked CPU
dependency set. Install [uv](https://docs.astral.sh/uv/) and run:

```powershell
git clone https://github.com/h7karu/regenerators_aigc_detector.git
cd regenerators_aigc_detector
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

Linux users can run:

```bash
git clone https://github.com/h7karu/regenerators_aigc_detector.git
cd regenerators_aigc_detector
bash scripts/setup_linux.sh
```

The setup scripts install the pinned interpreter, create `.venv`, synchronize
the locked CPU dependencies, construct the model, and verify local manifests
without loading the reserved test images. See [ARTIFACTS.md](ARTIFACTS.md) for
the separately distributed demonstration checkpoint and its checksum.

## Current status

Five model configurations are implemented: a Swin-Tiny RGB baseline, a robust
paired-view RGB model, an RGB-plus-Fourier-phase model with learned gated
fusion, a parameter-efficient LoRA variant, and the multi-source training
configuration. CIFAKE manifests, SID
streaming, WildFake metadata conversion, hierarchical sampling, leakage checks,
deterministic robustness evaluation, checkpointing, tests, and directory-to-JSON
inference are operational.

The saved smoke-test checkpoint is only an integration check. Its predictions
are not a trained detector and must not be reported as experimental results.
The local demonstration checkpoint is a separate three-epoch experiment on a
5,000-image CIFAKE training subset; it reached 0.9757 validation AUROC on 1,000
validation images, but it is not the final multi-source model.
The matching rank-4 LoRA experiment reached 0.9713 validation AUROC at epoch 1
while training 2,703,105 parameters. Neither result uses the held-out test set.

## Approach

The project is being developed in measurable stages:

1. Train a pretrained Swin-Tiny RGB baseline.
2. Train on clean/degraded image pairs with difficulty-aware consistency.
3. Fuse Swin RGB evidence with a lightweight Fourier-phase encoder.
4. Add parameter-efficient LoRA adaptation to Swin attention. (Implemented.)
5. Add generator-balanced WildFake and SID subsets. (Implemented.)
6. Compare clean, transformed, and held-out-generator performance.

The RGB model contains about 27.5 million parameters and the phase-fusion model
contains about 30.1 million, both well below the hackathon's 2-billion limit.

## Repository layout

```text
configs/             Experiment configurations
scripts/             Dataset preparation commands
tests/               Automated data/model/loss tests
requirements-cpu.lock Hash-locked Windows CPU/demo environment
requirements-cpu-linux.lock Hash-locked Linux CPU/demo environment
ARTIFACTS.md          External checkpoint locations and checksums
augmentations.py     Training and benchmark transformations
datasets.py          Manifest-backed PyTorch dataset
evaluate.py          Clean and robustness evaluation
losses.py            Classification and consistency objectives
manifest_schema.py   Canonical metadata and leakage validation
metrics.py           AUROC, AP, F1, FPR, and related metrics
model.py             RGB and gated RGB-phase detectors
predict.py           Required directory-to-JSON inference
train.py             Training and checkpoint selection
utils.py             Paths, configuration, seeding, and JSON output
```

## Environment and reproducibility

The supported interpreter is pinned in `.python-version`. Direct dependencies
are pinned in `requirements.txt`, `requirements-demo.txt`, and
`requirements-cpu.txt`; the resolved Windows CPU environment, including wheel
hashes, is committed as `requirements-cpu.lock`. Linux uses the separately
resolved `requirements-cpu-linux.lock`; binary locks are not assumed to be
portable between operating systems.

After running the quick-start script, select this VS Code interpreter:

```text
.venv\Scripts\python.exe
```

Verify an existing installation at any time without evaluating test images:

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
.\.venv\Scripts\python.exe -m pip check
```

For NVIDIA training, create a fresh Python 3.12.14 environment, install the
CUDA-specific `torch==2.13.0` and `torchvision==0.28.0` wheels recommended by
PyTorch for that machine, then install `requirements.txt`. Do not install the
CPU lock into a CUDA training environment.

When direct dependencies change intentionally, regenerate the Windows CPU lock
from the repository root:

```powershell
uv pip compile requirements-cpu.txt `
  --output-file requirements-cpu.lock `
  --python-version 3.12.14 `
  --python-platform windows `
  --index-strategy unsafe-best-match `
  --generate-hashes
```

Generate the Linux lock with the same command after replacing
`--python-platform windows` with `--python-platform linux` and selecting
`requirements-cpu-linux.lock` as the output file. macOS users should create a
Python 3.12.14 environment and install the platform-appropriate PyTorch build
before installing `requirements-demo.txt`; a macOS lock has not been verified.

## Prepare CIFAKE

Place or extract CIFAKE at:

```text
cifake-real-and-ai-generated-synthetic-images/
├── train/
│   ├── FAKE/
│   └── REAL/
└── test/
    ├── FAKE/
    └── REAL/
```

Build manifests:

```powershell
python scripts/build_manifest.py
```

This produces a stratified split:

| Split | Authentic | Generated |
|---|---:|---:|
| Train | 45,000 | 45,000 |
| Validation | 5,000 | 5,000 |
| Reserved test | 10,000 | 10,000 |

To decode every image and compute hashes during an extended data audit:

```powershell
python scripts/build_manifest.py --verify-images --hash
```

## Prepare SID-Set and WildFake

SID-Set is streamed from Hugging Face so a bounded subset can be materialised
without downloading its roughly 140 GB release. SID's original labels are
retained (`0=real`, `1=full synthetic`, `2=tampered`), while labels 1 and 2 map
to the detector's binary positive class:

```powershell
python scripts/build_sid_subset.py `
  --split train `
  --real-count 5000 `
  --synthetic-count 2500 `
  --tampered-count 2500

python scripts/build_sid_subset.py `
  --split validation `
  --real-count 1000 `
  --synthetic-count 500 `
  --tampered-count 500
```

WildFake remains archive-backed. Download only the desired archives and the
publisher's official `train_metadata.csv` and `test_metadata.csv`, extract the
archives under one `Images` directory, then run:

```powershell
python scripts/build_wildfake_manifest.py `
  --train-metadata path\to\train_metadata.csv `
  --test-metadata path\to\test_metadata.csv `
  --images-root path\to\WildFake\Images `
  --max-per-group 2000 `
  --check-paths
```

The adapter preserves the publisher's official train/test membership and
carves validation only from official train records. Hackathon-reserved COCO
val2017 and DALL-E Advanced examples are removed and rejected by the shared
manifest validator.

Merge the available source manifests:

```powershell
python scripts/merge_manifests.py `
  --inputs `
    data/manifests/cifake_all.csv `
    data/manifests/sid_train.csv `
    data/manifests/sid_val.csv `
    data/manifests/wildfake_all.csv `
  --name combined `
  --check-paths
```

Every manifest keeps a binary label plus original label, dataset, generator
family, architecture, weight type, version, source, and content ID. Duplicate
paths and content identities crossing split boundaries fail validation.

## Run tests

```powershell
python -m pytest -q
```

The tests cover every named robustness transform, deterministic noise,
manifest loading, hierarchical sampling, split leakage and reserved-source
checks, WildFake metadata conversion, consistency loss, model output shape,
finite logits, and the parameter limit.
They do not run `evaluate.py` and do not score the reserved CIFAKE test split.

## Train the RGB baseline

```powershell
python train.py --config configs/rgb_baseline.yaml
```

Training writes:

```text
checkpoints/rgb_baseline_best.pt
checkpoints/last.pt
logs/rgb_baseline/
```

The selected checkpoint maximises validation AUROC. The published CIFAKE test
split is not used for early stopping or threshold selection.

## Train robust and phase models

Robust RGB training creates a clean and degraded view after applying identical
random crop/flip geometry. Both views receive classification supervision, and
the difficult pairs receive a stronger consistency penalty.

```powershell
python train.py --config configs/robust_rgb.yaml
```

The phase model reconstructs RGB values in `[0,1]`, computes a two-dimensional
FFT in float32, represents phase using sine and cosine channels, and fuses the
phase embedding with Swin features through a learned gate.

```powershell
python train.py --config configs/phase_robust.yaml
```

The resulting checkpoint names are:

```text
checkpoints/robust_rgb_best.pt
checkpoints/phase_robust_best.pt
```

### Train with LoRA

LoRA mode freezes the pretrained Swin weights and inserts rank-4 trainable
adapters into its attention query/key/value and output projections. The phase
encoder, fusion gate, and classifier remain trainable. This reduces the local
phase model from 30,116,475 trainable parameters under full fine-tuning to
2,703,105 trainable parameters (105,984 in the Swin LoRA adapters).

Run the CPU-sized 5,000-train/1,000-validation experiment with:

```powershell
python train.py --config configs/local_lora_experiment.yaml --device cpu
```

Outputs are isolated from the existing full-fine-tuning experiment:

```text
checkpoints/local_lora_experiment/local_lora_best.pt
checkpoints/local_lora_experiment/last.pt
logs/local_lora_experiment/
```

The configured modes are `full`, `frozen`, and `lora`. LoRA rank, scaling,
dropout, and target layer suffixes live under the configuration's `model`
section. The held-out test manifest is recorded for later evaluation but is
not loaded by `train.py`.

For multi-source training, follow [COLAB_TRAINING.md](COLAB_TRAINING.md), then
run the readiness gate and GPU runner:

```bash
python scripts/check_training_readiness.py \
  --config configs/multisource_phase_robust.yaml \
  --require-cuda
bash scripts/train_gpu.sh configs/multisource_phase_robust.yaml
```

For a quick integration check:

```powershell
python train.py `
  --config configs/rgb_baseline.yaml `
  --device cpu `
  --epochs 1 `
  --limit-train-samples 4 `
  --limit-val-samples 4 `
  --max-train-batches 1 `
  --max-val-batches 1 `
  --no-pretrained
```

## Launch the interactive demo

The demo loads the locally trained RGB-plus-phase checkpoint and provides two
views: a direct detector and a robustness comparison across JPEG compression,
blur, and downscaling. It does not read the reserved test manifest.

Install the optional UI dependencies:

```powershell
python -m pip install -r requirements-demo.txt
```

In a fresh clone, first obtain the external checkpoint described in
[ARTIFACTS.md](ARTIFACTS.md) and place it at the documented path.

On the current Windows development machine, use the environment-aware launcher:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1
```

Then open <http://127.0.0.1:7860>. The direct cross-platform command is:

```powershell
python demo.py `
  --checkpoint checkpoints/local_phase_experiment/local_phase_best.pt `
  --config configs/local_phase_experiment.yaml
```

Public sharing is disabled by default. Add `--share` only when a temporary
public Gradio link is intentionally required. Uploaded images are processed in
memory and are not written to the repository by the application.

For a hosted demonstration, open `notebooks/demo.ipynb` in Colab, store the
checkpoint in Google Drive, update the checkpoint path cell, and run the cells
from top to bottom. Both launchers call the same inference and UI code.

## Evaluate robustness

Evaluate the reserved test split on every hackathon transformation:

```powershell
python evaluate.py `
  --config configs/rgb_baseline.yaml `
  --checkpoint checkpoints/rgb_baseline_best.pt `
  --split test `
  --all-transforms `
  --output reports/metrics/rgb_baseline_test.json
```

Available deterministic transformations include JPEG qualities 90/70/50/30,
Gaussian blur 0.5/1.0/2.0, downscale factors 0.5/0.25, Gaussian noise
0.02/0.05/0.10, colour jitter, and an 80% centre crop.

## Generate submission predictions

```powershell
python predict.py `
  --input-dir path\to\images `
  --checkpoint checkpoints\rgb_baseline_best.pt `
  --config configs\rgb_baseline.yaml `
  --output predictions.json
```

Output schema:

```json
[
  {
    "image_path": "path/to/image.jpg",
    "pred": 0.8173
  }
]
```

`pred` is the model's probability that the image is AI-generated.

## Limitations

- CIFAKE is low-resolution and generator-limited; it is a pipeline bootstrap,
  not sufficient evidence of real-world generalisation.
- The current local PyTorch environment is CPU-only. Full training should use
  an NVIDIA GPU or a hosted GPU runtime.
- The RGB, consistency, and phase implementations have passed smoke tests but
  still require full GPU training and controlled ablation results.
- WildFake images and the full SID training subset have not been downloaded
  locally; only a three-image SID streaming smoke subset has been materialised.
- Full GPU training, calibration analysis, and the final error report remain.
- Pixel-only detection cannot guarantee provenance and should not be the sole
  basis for high-impact moderation decisions.

## Contributions

Add final team-member contributions before submission.
