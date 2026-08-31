import torch
import pytest
from torch import nn

from model import AIGCDetector, LoRALinear, count_parameters, load_checkpoint


def test_lora_linear_starts_as_frozen_base_layer() -> None:
    base = nn.Linear(12, 8)
    inputs = torch.randn(3, 12)
    expected = base(inputs).detach()
    layer = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)

    actual = layer(inputs)
    assert torch.equal(actual, expected)
    assert not any(parameter.requires_grad for parameter in layer.base.parameters())
    assert all(parameter.requires_grad for parameter in layer.lora_a.parameters())
    assert all(parameter.requires_grad for parameter in layer.lora_b.parameters())


def test_lora_mode_only_trains_adapters_and_detector_head() -> None:
    model = AIGCDetector(
        pretrained=False,
        use_phase=True,
        phase_base_channels=8,
        training_mode="lora",
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    )

    assert len(model.lora_layers) == 24
    assert all(
        parameter.requires_grad == ("lora_a" in name or "lora_b" in name)
        for name, parameter in model.backbone.named_parameters()
    )
    assert all(parameter.requires_grad for parameter in model.phase_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.fusion_gate.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert count_parameters(model, trainable_only=True) < count_parameters(model) // 10
    assert [group["name"] for group in model.parameter_groups(1e-4, 3e-4)] == [
        "backbone",
        "head",
    ]


def test_swin_baseline_forward_shape() -> None:
    model = AIGCDetector(pretrained=False)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.shape == (1,)
    assert torch.isfinite(output).all()
    assert count_parameters(model) < 2_000_000_000


def test_phase_model_forward_and_gate() -> None:
    model = AIGCDetector(
        pretrained=False,
        use_phase=True,
        phase_base_channels=8,
    )
    model.eval()
    images = torch.zeros(1, 3, 224, 224)
    with torch.inference_mode():
        phase = model.phase_map(images)
        logits, auxiliary = model(images, return_aux=True)
    assert phase.shape == (1, 6, 224, 224)
    assert torch.isfinite(phase).all()
    assert phase.min() >= -1.0
    assert phase.max() <= 1.0
    assert logits.shape == (1,)
    assert auxiliary["gate"].shape == (1, 768)
    assert torch.all((auxiliary["gate"] >= 0.0) & (auxiliary["gate"] <= 1.0))
    assert count_parameters(model) < 2_000_000_000


def test_load_checkpoint_reports_unfetched_lfs_pointer(tmp_path) -> None:
    pointer = tmp_path / "model.pt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0123456789abcdef\n"
        "size 142704560\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="git lfs pull"):
        load_checkpoint(nn.Linear(1, 1), pointer)
