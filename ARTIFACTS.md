# External artifacts

Large datasets and model checkpoints are deliberately excluded from Git. This
keeps the repository cloneable and prevents accidental redistribution of data.

## Demo checkpoint

Place the locally trained demonstration checkpoint at:

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
validation rows. It is suitable for demonstrating the pipeline, not for making
claims about unseen generators or real-world deployment.

## LoRA experiment checkpoint

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
it remains excluded from normal Git because it is a large generated artifact.

## Datasets

Dataset files are also excluded. Rebuild deterministic manifests with the
commands in `README.md`. Keep the reserved test split untouched until the final
model and decision threshold have been frozen.
