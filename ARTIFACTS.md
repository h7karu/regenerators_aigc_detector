# External artifacts

Large datasets and training checkpoints are deliberately excluded from normal
Git. The single deployed SID checkpoint is an explicit exception tracked with
Git LFS so a clone can retrieve the exact model used by the application.

## Deployed SID LoRA checkpoint

The checkpoint used by the default demo belongs at:

```text
checkpoints/sid_local_lora/sid_local_lora_best.pt
```

After a complete LFS-enabled clone, the relevant folder structure is:

```text
checkpoints/
└── sid_local_lora/
    └── sid_local_lora_best.pt
```

Expected properties for the current final artifact:

```text
Size:   142704560 bytes
SHA256: f71653e7321068a193685abfc43fb3accb6f05314335a15b62215fd7e135af43
```

It was initialized from the fully trained CIFAKE model and fine-tuned for all
six epochs in `configs/sid_local_lora.yaml`. Epoch 6 is the selected checkpoint,
with a mean seven-condition validation AUROC of 0.9451 and a worst-condition
AUROC of 0.9345. The deployed inference policy aggregates clean, JPEG-70,
blur-1.0, resize-0.5, and crop-0.8 logits with a trimmed mean and uses the
SID-validation-selected threshold of 0.4856. Its five-view model-selection
AUROC is 0.9484, with 0.8825 balanced accuracy and 0.8834 F1.

The cross-domain validation benchmark marginally selected a 95% SID / 5%
CIFAKE ensemble. The published default remains the standalone SID checkpoint
so inference requires only the single versioned artifact documented here. This
checkpoint is the only `.pt` file unignored by the repository and must be
published through Git LFS.

Repository owners publish it with:

```bash
git lfs install
git add .gitattributes .gitignore \
  checkpoints/sid_local_lora/sid_local_lora_best.pt
git commit -m "Track deployed SID checkpoint with Git LFS"
git push
```

Do not run `git add` on the checkpoint until Git LFS is installed; otherwise a
normal Git push will fail because the binary is too large.

## CIFAKE initialization checkpoint

The SID model was initialized from:

```text
checkpoints/full_cifake_lora/full_cifake_lora_best.pt
```

Expected properties:

```text
Size:   142703714 bytes
SHA256: 3380ae9ff1a00ac11db2e1de018517504c4fb48bc7ecdb39ea5e19e7a5266679
```

It completed all 15 epochs configured in `configs/full_cifake_lora.yaml` and
reached 0.9970 AUROC on the untouched 20,000-image CIFAKE test split. It is no
longer the default inference artifact.

## Legacy local phase experiment checkpoint

The earlier locally trained demonstration checkpoint belongs at:

```text
checkpoints/local_phase_experiment/local_phase_best.pt
```

Expected properties:

```text
Size:   361655887 bytes
SHA256: d6463a2c6620d9a7b42f55f032100eb6e6fa1decd9b48166e87a8b621b2891de
```

Verify it on Windows:

```powershell
Get-FileHash -Algorithm SHA256 `
  checkpoints\local_phase_experiment\local_phase_best.pt
```

Verify it on Linux or macOS:

```bash
sha256sum checkpoints/local_phase_experiment/local_phase_best.pt
```

The checkpoint is a three-epoch local experiment trained with
`configs/local_phase_experiment.yaml` on 5,000 CIFAKE training rows and 1,000
validation rows. It is retained for experiment reproducibility and is not used
by the current demo or inference defaults.

## Legacy LoRA experiment checkpoint

The rank-4 LoRA experiment writes its best validation checkpoint to:

```text
checkpoints/local_lora_experiment/local_lora_best.pt
```

Expected properties:

```text
Size:   142693068 bytes
SHA256: 7ce06ec62c446a74dd7751ae92b1762599559d9662a82615043325be712caea3
```

It was trained for three CPU epochs with
`configs/local_lora_experiment.yaml` on the same 5,000 CIFAKE training rows and
1,000 validation rows. The best checkpoint was selected at epoch 1 with 0.9713
validation AUROC. It contains the frozen pretrained backbone together with the
LoRA adapters, phase encoder, fusion gate, classifier, and training metadata;
it remains excluded from normal Git because it is a large generated artifact
and is not used by the current demo or inference defaults.

## Datasets

Dataset files are also excluded. Rebuild deterministic manifests with the
commands in `README.md`. Keep the reserved test split untouched until the final
model and decision threshold have been frozen.
