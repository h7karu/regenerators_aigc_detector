# Regenerators AIGC Detector

Detecting AI-generated and AI-tampered images from pixels alone.

Given an image, the detector returns a score for how likely it is to be
AI-generated. It addresses two cases together: fully synthetic images, and real
photographs whose contents have been locally edited or tampered. The trained
model ships with a Gradio web interface, a directory-to-JSON CLI, and a
reusable inference module.

The detector is research software. Its score is not a calibrated probability
or proof of image provenance.

## Contents

- [Approach](#approach)
- [Final model results](#final-model-results)
- [Run the trained model on another machine](#run-the-trained-model-on-another-machine)
  — setup and installation
- [Contributor setup](#contributor-setup)
- [Repository layout](#repository-layout)
- [Reproduce the reported results](#reproduce-the-reported-results)
  — the end-to-end training and evaluation path
- [Evaluate the frozen checkpoint](#evaluate-the-frozen-checkpoint)
- [Limitations and future work](#limitations-and-future-work)
- [Team member contributions](#team-member-contributions)

## Approach

The model fuses two complementary views of the same image:

- **RGB semantic branch** — a Swin-Tiny backbone
  (`swin_tiny_patch4_window7_224`) adapted with LoRA on its attention
  projections (`attn.qkv`, `attn.proj`). LoRA keeps the trained parameter count
  small, which is what made a two-stage fine-tune affordable on a single
  consumer GPU.
- **Fourier-phase branch** — a compact CNN over six-channel sine/cosine maps of
  the FFT phase spectrum. Generators leave periodic upsampling and resampling
  traces in the frequency domain that are not apparent in RGB space and that
  survive many local edits.

Training ran in two stages: initialization on CIFAKE, then fine-tuning on a
balanced 40,000-image SID-Set subset covering real, fully synthetic, and
tampered images. Each training step pairs a clean view with a degraded view
under a robustness loss, so the objective rewards agreement under JPEG, blur,
resize, and noise rather than accuracy on pristine images alone. Checkpoints
are selected by mean AUROC across a seven-condition validation policy, with
worst-condition AUROC breaking ties.

At inference, the default policy applies deterministic five-view test-time
augmentation and combines the resulting logits with a trimmed mean.

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

**[notebooks/results.ipynb](notebooks/results.ipynb)** walks through these
results with rendered plots: cross-domain transfer (why SID adaptation was
needed), robustness curves under JPEG/blur/resize/noise, per-generator accuracy
(fully-synthetic vs. locally-tampered images), the test-time-augmentation gain,
and the confident-error cases. It also carries a 200-image branch ablation from
an earlier CLIP-plus-forensic design, retained as a record of that exploration
rather than as a measurement of the deployed model.

The notebook loads no checkpoint and runs no inference — it reads the JSON and
CSV under `reports/` and finishes in seconds. Outputs are saved in the
notebook, so it renders directly on GitHub without being executed. Figures
whose inputs are absent from a given clone print a note and skip.

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
notebooks/              Hosted demo notebook and a rendered results walkthrough
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

## Reproduce the reported results

The numbers in [Final model results](#final-model-results) come from the
end-to-end path below. Steps 2 and 4 need an NVIDIA GPU; the evaluation in step
5 runs on CPU. The stage-2 SID fine-tune dominates the wall-clock cost.

| # | Step | Where | Produces |
| --- | --- | --- | --- |
| 1 | Build CIFAKE manifests | [Prepare CIFAKE for training](#prepare-cifake-for-training) | `data/manifests/cifake_*.csv` |
| 2 | Stage-1 CIFAKE training | [Train or resume the final configuration](#train-or-resume-the-final-configuration) | `checkpoints/full_cifake_lora/full_cifake_lora_best.pt` |
| 3 | Build the balanced SID subset and splits | [Faster local SID training](#faster-local-sid-training) | `data/manifests/sid_*.csv` |
| 4 | Stage-2 SID fine-tune | [Faster local SID training](#faster-local-sid-training) | `checkpoints/sid_local_lora/sid_local_lora_best.pt` |
| 5 | Robustness evaluation | [Evaluate the frozen checkpoint](#evaluate-the-frozen-checkpoint) | `reports/metrics/sid_local_lora_tta_test_robustness.json` |

Step 3 has an alternative: `configs/sid_streaming_lora.yaml` streams SID-Set
from Hugging Face without persisting images. See
[Stream SID-Set without storing images](#stream-sid-set-without-storing-images).

Each reported value traces to a committed artifact:

| Reported value | Artifact |
| --- | --- |
| Model-selection TTA row (0.9484 AUROC) and the 0.4856 threshold | `reports/metrics/inference_policy_validation.json` |
| Holdout clean row (0.9474 AUROC), the 0.9359 mean, and `blur_2.0` at 0.9049 | `reports/metrics/sid_local_lora_tta_test_robustness.json` |
| Single-view holdout comparison (0.9427 AUROC, no TTA) | `reports/metrics/sid_local_lora_test_robustness.json` |
| Pre-fine-tune transfer baselines | `reports/metrics/pre_sid_cifake_robustness.json`, `reports/metrics/pre_sid_sid_holdout_robustness.json` |

To read the results without retraining anything, open
[notebooks/results.ipynb](notebooks/results.ipynb), which uses these files
directly.

Seeds and split construction are deterministic, but exact floating-point values
can still shift across a different GPU, driver, or PyTorch build.

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

## Limitations and future work

### Limitations

- SID and CIFAKE cover a limited generator and manipulation distribution. Both
  stages saw a fixed set of generators, so nothing here demonstrates transfer
  to architectures released after that data was collected.
- Generalisation to arbitrary generators, edits, screenshots, and social-media
  processing has not been established.
- The validation-selected threshold may need recalibration for other data. The
  output is a decision score, not a calibrated probability.
- `blur_2.0` is the weakest condition at 0.9049 AUROC, and `resize_0.25` is
  second at 0.9181. Both destroy the high-frequency evidence the phase branch
  depends on, which is a structural weakness of the design rather than a
  tuning problem.
- On the SID holdout, 15.8% of real images are still flagged at the deployed
  threshold (89.4% of AI images are caught). Wherever a false accusation is
  costly, that false-positive rate is the binding constraint, not AUROC.
- Pixel-only detection is supporting evidence, not definitive provenance. The
  model cannot see C2PA signatures, EXIF, or distribution context.

### What we would improve with more time

- **Wider generator coverage.** Train and evaluate across additional sources —
  WildFake, and diffusion families absent from SID — to measure the
  cross-generator gap directly instead of assuming it.
  `scripts/build_wildfake_manifest.py` and
  `configs/multisource_phase_robust.yaml` are the groundwork already in place.
- **Calibration.** Fit temperature scaling or isotonic regression on a held-out
  split so the score reads as a probability, and report expected calibration
  error next to AUROC.
- **Localization.** SID-Set ships tampering masks that the current pipeline
  discards before decoding. Predicting *where* an image was edited would be far
  more actionable than a single image-level score, and the masks to supervise
  it are already there.
- **Ablate the deployed architecture.** The ablation artifacts in the results
  notebook probe an earlier CLIP-plus-forensic design at 200 images. An
  RGB-only / phase-only / fused comparison on the deployed model at the full
  2,000-image evaluation size would establish what the phase branch actually
  contributes.
- **Blur and downscale robustness.** Add scale-aware augmentation and
  multi-resolution inference to recover the loss at `blur_2.0` and
  `resize_0.25`.
- **Inference cost.** Five-view TTA multiplies cost by five for a modest gain
  (0.9427 to 0.9474 AUROC on the holdout). Distilling the TTA ensemble into a
  single forward pass would make deployment substantially cheaper.

## Team member contributions

| Member | Contribution |
| --- | --- |
| Zechary Chua | _TBD_ |
| Isaac Teo | _TBD_ |
| Wei Tianyue | _TBD_ |
| Tan Jie En, Nigel | _TBD_ |
| Kawaguchi Hikaru | _Refined the final UI, recorded the live demo and wrote the 
robustness evaluation summary and error analysis note. Aided in research and 
implementing other models to compare against._ |

## Contributing

Keep generated data and training checkpoints out of Git. The canonical deployed
SID checkpoint is the only Git-LFS-managed exception. Add tests for behavioural
changes and run `python -m pytest -q` before opening a pull request.
