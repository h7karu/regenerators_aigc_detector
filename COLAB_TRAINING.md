# GPU and Colab training runbook

The local machine has a CPU-only PyTorch installation. Use an NVIDIA runtime
for full experiments; keep local runs limited to tests and integration smoke
checks.

## 1. Start the runtime

In Colab, select **Runtime > Change runtime type > T4 GPU** (or a stronger GPU),
then confirm it is visible:

```bash
!nvidia-smi
```

Clone the repository with Git LFS so the deployed SID checkpoint is reproduced
at the same path used by the application:

```bash
!apt-get update -qq
!apt-get install -y -qq git-lfs
!git lfs install
!git clone https://github.com/h7karu/regenerators_aigc_detector.git \
  /content/regenerators_aigc_detector
%cd /content/regenerators_aigc_detector
!git lfs pull \
  --include="checkpoints/sid_local_lora/sid_local_lora_best.pt"
```

Do not commit datasets, credentials, or newly generated training checkpoints.
The deployed SID checkpoint is the sole repository-managed exception. Commands
below that initialize training from the older CIFAKE checkpoint still require
that separate artifact at the path documented in `ARTIFACTS.md`.

## 2. Install dependencies

Colab already provides a CUDA-enabled PyTorch build. Keep it, then install the
remaining pinned packages:

```bash
%cd /content/regenerators_aigc_detector
!python -m pip install -r requirements.txt
```

If pip attempts to replace Colab's CUDA PyTorch, install only the packages
listed after the PyTorch comments in `requirements.txt`.

## 3. Materialise bounded dataset subsets

For machines with limited storage, skip SID materialisation and use the
streaming configuration instead. First run the bounded preflight:

```bash
!python scripts/check_training_readiness.py \
  --config configs/sid_streaming_smoke.yaml \
  --require-cuda

!python train.py \
  --config configs/sid_streaming_smoke.yaml \
  --device cuda \
  --batch-size 32 \
  --limit-train-samples 2048 \
  --limit-val-samples 512 \
  --init-checkpoint checkpoints/full_cifake_lora/full_cifake_lora_best.pt
```

Only after that succeeds, start the full stream:

```bash
!python train.py \
  --config configs/sid_streaming_lora.yaml \
  --device cuda \
  --init-checkpoint checkpoints/full_cifake_lora/full_cifake_lora_best.pt
```

This stores no SID images locally. It still transfers the image stream again
for every epoch, so materialisation is preferable when disk is available and
repeated experiments are planned. The full configuration uses seven validation
conditions with 2,000 samples each and selects checkpoints by mean condition
AUROC rather than clean performance alone. Paired views make its configured
batch of 128 equivalent to 256 images in the model forward pass. This measured
17.8 GiB reserved on an RTX 3090; lower it to 64 or 32 on smaller GPUs.

If remote streaming is slower than the GPU, materialise the balanced subset
shown below and train `configs/sid_local_lora.yaml`. It uses four persistent
workers and prefetches four batches per worker. On the measured host, local
paired-view loading reached about 1,257 source images/s while the RTX 3090
trained at about 281 source images/s, leaving enough loader headroom to keep the
GPU occupied.

Start with a 10,000-image SID subset. The quotas keep the binary labels balanced
while retaining both full-synthetic and tampered positives:

```bash
!python scripts/build_sid_subset.py \
  --split train \
  --real-count 5000 \
  --synthetic-count 2500 \
  --tampered-count 2500

!python scripts/build_sid_subset.py \
  --split validation \
  --real-count 1000 \
  --synthetic-count 500 \
  --tampered-count 500
```

For WildFake, download only selected archives plus the publisher's official
`train_metadata.csv` and `test_metadata.csv`. After extraction, convert them:

```bash
!python scripts/build_wildfake_manifest.py \
  --train-metadata /content/WildFake/train_metadata.csv \
  --test-metadata /content/WildFake/test_metadata.csv \
  --images-root /content/WildFake/Images \
  --max-per-group 2000 \
  --check-paths
```

The adapter preserves the publisher's train/test membership, carves validation
only from the official train split, and removes the hackathon-reserved COCO
val2017 and DALL-E Advanced sources.

## 4. Merge manifests

Merge only manifests that exist in the runtime. A typical command is:

```bash
!python scripts/merge_manifests.py \
  --inputs \
    data/manifests/cifake_all.csv \
    data/manifests/sid_train.csv \
    data/manifests/sid_val.csv \
    data/manifests/wildfake_all.csv \
  --name combined \
  --check-paths
```

The merge fails on duplicate paths, content IDs crossing split boundaries,
invalid labels, missing images, or reserved demonstration data.

## 5. Run the readiness gate and train

```bash
!python scripts/check_training_readiness.py \
  --config configs/multisource_phase_robust.yaml \
  --require-cuda

!bash scripts/train_gpu.sh configs/multisource_phase_robust.yaml
```

The sampler first balances real versus generated/manipulated labels, then
datasets within each label, then generators within each label/dataset branch.
This prevents a large generator or dataset from dominating an epoch.

## 6. Preserve outputs and evaluate

Copy `checkpoints/`, `logs/`, and metric JSON files to persistent storage. Run
the clean and degradation suite on the untouched combined test split:

```bash
!python evaluate.py \
  --config configs/multisource_phase_robust.yaml \
  --checkpoint checkpoints/multisource_phase_robust_best.pt \
  --split test \
  --all-transforms \
  --output reports/metrics/multisource_phase_robust_test.json
```

Do not use test metrics to choose a checkpoint or probability threshold.
