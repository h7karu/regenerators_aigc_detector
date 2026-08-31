"""Model definitions for the Regenerators AIGC detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import timm
import torch
from torch import nn

from utils import resolve_project_path


class LoRALinear(nn.Module):
    """Low-rank update around a frozen linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1).")
        self.base = base
        self.base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.lora_b(self.lora_a(self.dropout(inputs)))
        return self.base(inputs) + update * self.scaling


def inject_lora(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    targets: tuple[str, ...],
) -> tuple[str, ...]:
    """Replace matching linear layers and return their fully qualified names."""

    if not targets:
        raise ValueError("At least one LoRA target must be configured.")
    replacements: list[tuple[str, nn.Linear]] = []
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and any(name.endswith(target) for target in targets):
            replacements.append((name, child))
    if not replacements:
        raise ValueError(f"No linear layers matched LoRA targets: {targets}")

    replaced_names: list[str] = []
    for name, child in replacements:
        parent_name, _, attribute = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(
            parent,
            attribute,
            LoRALinear(
                child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            ),
        )
        replaced_names.append(name)
    return tuple(replaced_names)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        )
        self.activation = nn.GELU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(inputs) + self.shortcut(inputs))


class PhaseEncoder(nn.Module):
    """Compact CNN for six-channel sine/cosine Fourier-phase maps."""

    def __init__(self, output_dim: int, base_channels: int = 32) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(6, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            ResidualConvBlock(base_channels, base_channels * 2, stride=2),
            ResidualConvBlock(base_channels * 2, base_channels * 4, stride=2),
            ResidualConvBlock(base_channels * 4, base_channels * 8, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 8, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(phase))


class AIGCDetector(nn.Module):
    """Pretrained visual backbone with a calibrated binary classification head."""

    def __init__(
        self,
        backbone: str = "swin_tiny_patch4_window7_224.ms_in1k",
        *,
        pretrained: bool = True,
        dropout: float = 0.2,
        cache_dir: str | Path = "models",
        use_phase: bool = False,
        phase_base_channels: int = 32,
        training_mode: str = "full",
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.05,
        lora_targets: tuple[str, ...] = ("attn.qkv", "attn.proj"),
    ) -> None:
        super().__init__()
        training_mode = training_mode.lower()
        if training_mode not in {"full", "frozen", "lora"}:
            raise ValueError(
                "training_mode must be one of: full, frozen, lora"
            )
        self.backbone_name = backbone
        self.training_mode = training_mode
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            cache_dir=str(resolve_project_path(cache_dir)),
        )
        self.lora_layers: tuple[str, ...] = ()
        if training_mode in {"frozen", "lora"}:
            self.backbone.requires_grad_(False)
        if training_mode == "lora":
            self.lora_layers = inject_lora(
                self.backbone,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                targets=lora_targets,
            )
        feature_dim = int(self.backbone.num_features)
        self.use_phase = use_phase
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )
        if use_phase:
            self.phase_encoder = PhaseEncoder(feature_dim, phase_base_channels)
            self.fusion_gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid(),
            )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def train(self, mode: bool = True) -> AIGCDetector:
        super().train(mode)
        if self.training_mode == "frozen":
            self.backbone.eval()
        return self

    def phase_map(self, normalized_images: torch.Tensor) -> torch.Tensor:
        """Recover RGB [0,1], then encode circular FFT phase as sine/cosine."""
        with torch.autocast(device_type=normalized_images.device.type, enabled=False):
            images = (
                normalized_images.float() * self.image_std + self.image_mean
            ).clamp(0.0, 1.0)
            spectrum = torch.fft.fft2(images, dim=(-2, -1), norm="ortho")
            spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
            phase = torch.angle(spectrum)
            return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)

    def forward_features(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rgb_features = self.backbone(images)
        if not self.use_phase:
            return rgb_features, None
        phase_features = self.phase_encoder(self.phase_map(images))
        gate = self.fusion_gate(torch.cat([rgb_features, phase_features], dim=-1))
        fused_features = gate * rgb_features + (1.0 - gate) * phase_features
        return fused_features, gate

    def forward(
        self, images: torch.Tensor, *, return_aux: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features, gate = self.forward_features(images)
        logits = self.classifier(features).squeeze(-1)
        if return_aux:
            auxiliary = {}
            if gate is not None:
                auxiliary["gate"] = gate
            return logits, auxiliary
        return logits

    def parameter_groups(
        self, backbone_lr: float, head_lr: float
    ) -> list[dict[str, Any]]:
        head_parameters = list(self.classifier.parameters())
        if self.use_phase:
            head_parameters.extend(self.phase_encoder.parameters())
            head_parameters.extend(self.fusion_gate.parameters())
        backbone_parameters = [
            parameter for parameter in self.backbone.parameters() if parameter.requires_grad
        ]
        head_parameters = [
            parameter for parameter in head_parameters if parameter.requires_grad
        ]
        groups: list[dict[str, Any]] = []
        if backbone_parameters:
            groups.append(
                {"params": backbone_parameters, "lr": backbone_lr, "name": "backbone"}
            )
        if head_parameters:
            groups.append({"params": head_parameters, "lr": head_lr, "name": "head"})
        if not groups:
            raise ValueError("The model has no trainable parameters.")
        return groups


def build_model(config: dict[str, Any], *, pretrained: bool | None = None) -> AIGCDetector:
    model_config = config["model"]
    use_pretrained = (
        bool(model_config.get("pretrained", True))
        if pretrained is None
        else pretrained
    )
    return AIGCDetector(
        backbone=model_config["backbone"],
        pretrained=use_pretrained,
        dropout=float(model_config.get("dropout", 0.2)),
        cache_dir=model_config.get("cache_dir", "models"),
        use_phase=bool(model_config.get("use_phase", False)),
        phase_base_channels=int(model_config.get("phase_base_channels", 32)),
        training_mode=str(model_config.get("training_mode", "full")),
        lora_rank=int(model_config.get("lora_rank", 4)),
        lora_alpha=float(model_config.get("lora_alpha", 8.0)),
        lora_dropout=float(model_config.get("lora_dropout", 0.05)),
        lora_targets=tuple(
            model_config.get("lora_targets", ["attn.qkv", "attn.proj"])
        ),
    )


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    path = resolve_project_path(checkpoint_path)
    if path.is_file() and path.stat().st_size < 1024:
        header = path.read_bytes()[:200]
        if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError(
                f"Checkpoint is only a Git LFS pointer: {path}. "
                "Install Git LFS and run `git lfs pull --include="
                '"checkpoints/sid_local_lora/sid_local_lora_best.pt"`.'
            )
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)
