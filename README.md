# Regenerators AIGC Detector

An experimental image-forensics detector built with a LoRA-adapted Swin-Tiny
RGB backbone and a learned Fourier-phase branch. The final CIFAKE model was
trained on 90,000 images and selected on a separate 10,000-image validation
split.

The detector is research software. Its score is not a calibrated probability
or proof of image provenance.

## Final model results

| Split | Images | AUROC | Balanced accuracy | F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 10,000 | 0.9970 | 0.9741 | 0.9742 |
| Held-out test | 20,000 | 0.9970 | 0.9743 | 0.9744 |

Configuration: `configs/full_cifake_lora.yaml`

Checkpoint: `checkpoints/full_cifake_lora/full_cifake_lora_best.pt`

Large checkpoints and datasets are excluded from Git. See
[ARTIFACTS.md](ARTIFACTS.md) for the checkpoint checksum.

## Run the trained model on another machine

Training data is not required for inference.

### 1. Clone the repository

```bash
git clone https://github.com/h7karu/regenerators_aigc_detector.git
cd regenerators_aigc_detector
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

For CPU-only inference on Linux or Windows:

```bash
python -m pip install torch==2.13.0+cpu torchvision==0.28.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu
```

For CPU inference on macOS:

```bash
python -m pip install torch==2.13.0 torchvision==0.28.0
```

For NVIDIA inference, install the matching CUDA build using the command from
the [PyTorch installation selector](https://pytorch.org/get-started/locally/),
then verify it:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

It should print `True` and the GPU name. Skip that check on a CPU-only machine.

### 4. Install project dependencies

```bash
python -m pip install -r requirements.txt
python -m pip check
```

There is intentionally one requirements file. PyTorch remains separate because
its CPU and CUDA wheels are platform-specific.

### 5. Download the checkpoint

Download `full_cifake_lora_best.pt` from the project's release assets or model
storage and place it at:

```text
checkpoints/full_cifake_lora/full_cifake_lora_best.pt
```

Verify it on Linux or macOS:

```bash
sha256sum checkpoints/full_cifake_lora/full_cifake_lora_best.pt
```

Or on Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 `
  checkpoints\full_cifake_lora\full_cifake_lora_best.pt
```

Expected SHA256:

```text
3380ae9ff1a00ac11db2e1de018517504c4fb48bc7ecdb39ea5e19e7a5266679
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

### 7. Optional: run directory inference

```bash
python predict.py \
  --input-dir path/to/images \
  --checkpoint checkpoints/full_cifake_lora/full_cifake_lora_best.pt \
  --config configs/full_cifake_lora.yaml \
  --output predictions.json
```

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
model.py                Swin, LoRA, phase encoder, and fusion model
datasets.py             Manifest-backed datasets and sampling
augmentations.py        Training and benchmark transformations
metrics.py              AUROC, AP, F1, and related metrics
```

Generated `data/`, `checkpoints/`, `logs/`, `models/`, and `reports/`
directories are ignored by Git.

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

The run saves `last.pt` after every epoch and selects the best checkpoint by
validation AUROC. A checkpoint that completed every configured epoch is rejected
by the resume guard. For SID-Set, WildFake, multi-source preparation, and Colab,
see [COLAB_TRAINING.md](COLAB_TRAINING.md).

## Evaluate the frozen checkpoint

```bash
python evaluate.py \
  --config configs/full_cifake_lora.yaml \
  --checkpoint checkpoints/full_cifake_lora/full_cifake_lora_best.pt \
  --split test \
  --transform clean \
  --output reports/metrics/full_cifake_lora_test_clean.json
```

Replace `--transform clean` with `--all-transforms` for the degradation suite.

## Limitations

- CIFAKE is low-resolution and covers a limited generator distribution.
- Generalisation to arbitrary generators, edits, screenshots, and social-media
  processing has not been established.
- The validation-selected threshold may need recalibration for other data.
- Pixel-only detection is supporting evidence, not definitive provenance.

## Contributing

Keep generated data and checkpoints out of Git, add tests for behavioural
changes, and run `python -m pytest -q` before opening a pull request.
