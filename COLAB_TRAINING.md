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

Move the repository into the runtime, either from a private Git remote or from
Google Drive. Do not commit datasets, checkpoints, or credentials.

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
