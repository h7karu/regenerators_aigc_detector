# Regenerators AIGC Detector

An experimental image-forensics detector built with a LoRA-adapted Swin-Tiny
RGB backbone and a learned Fourier-phase branch. The deployed model was
initialized on CIFAKE, then fine-tuned on a balanced 40,000-image SID subset
covering real, fully synthetic, and tampered images.

The detector is research software. Its score is not a calibrated probability
or proof of image provenance.

## Final model results

| Split | Images | AUROC | Balanced accuracy | F1 |
| --- | ---: | ---: | ---: | ---: |
| SID model-selection (five-view TTA) | 2,000 | 0.9484 | 0.8825 | 0.8834 |
| SID holdout (clean) | 2,000 | 0.9474 | 0.8680 | 0.8713 |

These deployment results use deterministic five-view test-time augmentation
(`clean`, `jpeg_70`, `blur_1.0`, `resize_0.5`, and `crop_0.8`) with a trimmed
mean of model logits. Across the complete held-out degradation suite, mean
AUROC is 0.9359 and the worst condition is `blur_2.0` at 0.9049 AUROC. See
`reports/metrics/sid_local_lora_tta_test_robustness.json` for all conditions and
`reports/metrics/inference_policy_validation.json` for policy selection.

Configuration: `configs/sid_local_lora.yaml`

Checkpoint: `checkpoints/sid_local_lora/sid_local_lora_best.pt`

Training checkpoints and datasets are excluded from Git. The deployed SID
checkpoint is tracked through Git LFS; see [ARTIFACTS.md](ARTIFACTS.md) for its
path and checksum.

## Run the trained model on another machine

Training data is not required for inference.

### 1. Install Git LFS and clone the complete repository

Install [Git LFS](https://git-lfs.com/) before cloning. Then run:

```bash
git lfs install
git clone https://github.com/h7karu/regenerators_aigc_detector.git
cd regenerators_aigc_detector
git lfs pull --include="checkpoints/sid_local_lora/sid_local_lora_best.pt"
```

The resulting deployment folders must look like this:

```text
regenerators_aigc_detector/
├── checkpoints/
│   └── sid_local_lora/
│       └── sid_local_lora_best.pt
├── configs/
│   └── sid_local_lora.yaml
├── deployment.py
├── demo.py
└── requirements.txt
```

On Windows PowerShell, the same Git commands apply. Confirm the model exists:

```powershell
Test-Path checkpoints\sid_local_lora\sid_local_lora_best.pt
```

Install Python 3.12.14, the project's documented target.

### 2. Create and activate an environment

Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install PyTorch for the machine

For CPU-only inference on Windows PowerShell, use this as one line (PowerShell
does not use `\` for line continuation):

```powershell
python -m pip install torch==2.13.0+cpu torchvision==0.28.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

For CPU-only inference on Linux:

```bash
python -m pip install torch==2.13.0+cpu torchvision==0.28.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

For CPU inference on macOS:

```bash
python -m pip install torch==2.13.0 torchvision==0.28.0
```

For NVIDIA inference, do not use the CPU command above. Install the matching
CUDA build using the command from the
[PyTorch installation selector](https://pytorch.org/get-started/locally/), then
verify it:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`CUDA available` should be `True` and `Device` should show the GPU name. If it
prints `False`, confirm that the NVIDIA driver works and that the selector
command installed a CUDA build rather than the CPU build.

On every platform, confirm that both packages import and that their versions
match:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

For the pinned CPU build this prints versions beginning with `2.13.0+cpu` and
`0.28.0+cpu`.

If Windows reports `No matching distribution found`, confirm that the active
interpreter is the documented 64-bit Python 3.12 environment:

```powershell
python -c "import platform, sys; print(sys.executable); print(sys.version); print(platform.architecture()[0])"
```

It should point inside `.venv`, report Python 3.12.14, and print `64bit`. Do not
run the Linux command containing `\` in PowerShell.

### 4. Install project dependencies

```bash
python -m pip install -r requirements.txt
python -m pip check
```

There is intentionally one requirements file. PyTorch remains separate because
its CPU and CUDA wheels are platform-specific.

### 5. Verify the Git LFS checkpoint

The deployed `sid_local_lora_best.pt` checkpoint is versioned at the following
path through Git LFS rather than ordinary Git:

```text
checkpoints/sid_local_lora/sid_local_lora_best.pt
```

If the clone did not download the binary automatically, run these commands from
the repository root:

```bash
git lfs install
git lfs pull --include="checkpoints/sid_local_lora/sid_local_lora_best.pt"
```

Confirm that the downloaded file is approximately 143 MB. If it is only a
small text file beginning with `version https://git-lfs.github.com/spec/v1`, the
LFS object has not been downloaded yet; rerun `git lfs pull`.

Verify it on Linux or macOS:

```bash
sha256sum checkpoints/sid_local_lora/sid_local_lora_best.pt
```

Or on Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 `
  checkpoints\sid_local_lora\sid_local_lora_best.pt
```

Expected SHA256:

```text
f71653e7321068a193685abfc43fb3accb6f05314335a15b62215fd7e135af43
```

### 6. Start the web interface

The launcher defaults point to the final checkpoint and config:

```bash
python demo.py --device auto --host 127.0.0.1 --port 7860
```

Open <http://127.0.0.1:7860>. Use `--device cpu` or `--device cuda` to require a
specific device. For a remote machine, forward the port instead of exposing the
development server publicly:

```bash
ssh -L 7860:127.0.0.1:7860 user@server
```

The Gradio **Analyse image** action calls the canonical SID checkpoint through
the same five-view trimmed-mean TTA policy used by `predict.py` and
`evaluate.py`. Gradio also exposes this action as the `/analyse` API endpoint;
the robustness lab is available as `/robustness`.

### 7. Optional: run directory inference

```bash
python predict.py \
  --input-dir path/to/images \
  --output predictions.json
```

The prediction CLI shares the same SID checkpoint and configuration defaults as
the web interface, including five-view TTA and its validation-selected 0.4856
decision threshold. Pass `--single-view` to benchmark the raw checkpoint.

## Contributor setup

The setup scripts create a Python 3.12.14 CPU environment, install the single
requirements file, and verify it.

Linux:

```bash
bash scripts/setup_linux.sh
source .venv/bin/activate
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
```

Run tests with:

```bash
python -m pytest -q
```

## Repository layout

```text
configs/                Training and inference configurations
notebooks/              Hosted demo notebook
scripts/                Setup, data preparation, and GPU helpers
tests/                  Automated tests
ARTIFACTS.md            External artifact locations and checksums
COLAB_TRAINING.md        Multi-source GPU training runbook
requirements.txt        All direct project dependencies except PyTorch
train.py                Training and checkpoint selection
evaluate.py             Clean and robustness evaluation
predict.py              Directory-to-JSON inference
demo.py                 Gradio frontend launcher
demo_app.py             Frontend components and event handling
demo_inference.py       Reusable inference service
deployment.py           Canonical deployed checkpoint and config defaults
inference_policy.py      TTA and calibrated ensemble aggregation
model.py                Swin, LoRA, phase encoder, and fusion model
data_pipeline.py        Manifest and streaming datasets, plus sampling
augmentations.py        Training and benchmark transformations
metrics.py              AUROC, AP, F1, and related metrics
```

Generated `data/`, `logs/`, `models/`, and training-checkpoint files are ignored
by Git. The deployed SID checkpoint is the sole LFS-managed checkpoint
exception. Selected reproducible metric reports under `reports/metrics/` are
versioned with the source.

## Prepare CIFAKE for training

Download and extract CIFAKE into:

```text
cifake-real-and-ai-generated-synthetic-images/
├── train/
│   ├── FAKE/
│   └── REAL/
└── test/
    ├── FAKE/
    └── REAL/
```

Then build deterministic manifests:

```bash
bash scripts/download_cifake.sh
python scripts/build_manifest.py
```

Do not use the held-out test split for checkpoint or threshold selection.

## Train or resume the final configuration

Check readiness:

```bash
python scripts/check_training_readiness.py \
  --config configs/full_cifake_lora.yaml --require-cuda
```

Start training:

```bash
python train.py --config configs/full_cifake_lora.yaml --device cuda
```

Resume an interrupted run:

```bash
python train.py \
  --config configs/full_cifake_lora.yaml \
  --device cuda \
  --resume checkpoints/full_cifake_lora/last.pt
```

The run saves `last.pt` after every epoch. Configurations without an explicit
validation transform list select the best checkpoint by clean validation AUROC.
A checkpoint that completed every configured epoch is rejected by the resume
guard. For SID-Set, WildFake, multi-source preparation, and Colab, see
[COLAB_TRAINING.md](COLAB_TRAINING.md).

### Stream SID-Set without storing images

`configs/sid_streaming_lora.yaml` streams SID-Set directly from Hugging Face,
drops localization masks before decoding, and emits 50% real, 25% fully
synthetic, and 25% tampered examples. It validates on 2,000 fixed samples under
clean, JPEG, blur, resize, noise, colour, and crop conditions. The mean
per-condition AUROC selects the checkpoint; worst-condition AUROC breaks ties,
and one threshold is fitted to the pooled validation predictions.

Initialize it from the CIFAKE model:

```bash
python train.py \
  --config configs/sid_streaming_lora.yaml \
  --device cuda \
  --init-checkpoint checkpoints/full_cifake_lora/full_cifake_lora_best.pt
```

No SID images are persisted, but each epoch still consumes substantial network
bandwidth. Paired clean/degraded training doubles the forward-pass image count,
so the full configuration uses a source-image batch size of 128 on a measured
24 GiB RTX 3090. Resume an interrupted run with:

```bash
python train.py \
  --config configs/sid_streaming_lora.yaml \
  --device cuda \
  --resume checkpoints/sid_streaming_lora/last.pt
```

### Faster local SID training

SID-Set is approximately 130 GiB in its entirety. On slower Hugging Face
connections, repeatedly streaming remote Parquet shards leaves the GPU idle.
The recommended fast path is to authenticate with the Hugging Face CLI and
materialise a balanced bounded subset once. The builder discards localization
masks before decoding samples:

```bash
hf auth login

python scripts/build_sid_subset.py \
  --split train \
  --real-count 20000 \
  --synthetic-count 10000 \
  --tampered-count 10000

python scripts/build_sid_subset.py \
  --split validation \
  --real-count 2000 \
  --synthetic-count 1000 \
  --tampered-count 1000

python scripts/split_sid_validation.py
```

The final command creates a stratified 2,000-image model-selection split and a
separate 2,000-image holdout. Then train with the local, four-worker
configuration:

```bash
python scripts/check_training_readiness.py \
  --config configs/sid_local_lora.yaml \
  --require-cuda

python train.py \
  --config configs/sid_local_lora.yaml \
  --device cuda \
  --init-checkpoint checkpoints/full_cifake_lora/full_cifake_lora_best.pt
```

The local configuration enables persistent workers, four-batch prefetching,
cuDNN autotuning, TF32 for eligible float32 kernels, fused AdamW, AMP, and a
source batch size of 128. It retains the same paired-view robustness loss and
seven-condition validation policy as the streaming configuration.

## Evaluate the frozen checkpoint

```bash
python evaluate.py \
  --split test \
  --all-transforms \
  --output reports/metrics/sid_local_lora_tta_test_robustness.json
```

The evaluator shares the deployed SID defaults and TTA policy. Use
`--transform clean` instead of `--all-transforms` for a quick clean-only
evaluation, or `--single-view` to measure the checkpoint without TTA.

## Limitations

- SID and CIFAKE still cover a limited generator and manipulation distribution.
- Generalisation to arbitrary generators, edits, screenshots, and social-media
  processing has not been established.
- The validation-selected threshold may need recalibration for other data.
- Pixel-only detection is supporting evidence, not definitive provenance.

## Contributing

Keep generated data and training checkpoints out of Git. The canonical deployed
SID checkpoint is the only Git-LFS-managed exception. Add tests for behavioural
changes and run `python -m pytest -q` before opening a pull request.
