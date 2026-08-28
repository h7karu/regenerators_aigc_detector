"""Loss functions used by baseline and robustness experiments."""

from __future__ import annotations

import torch
from torch import nn


def binary_classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(
        logits.float(), labels.float()
    )


def difficulty_aware_consistency_loss(
    clean_logits: torch.Tensor,
    degraded_logits: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Prioritize view pairs whose AIGC probabilities disagree most."""
    clean_probability = clean_logits.detach().sigmoid()
    degraded_probability = degraded_logits.detach().sigmoid()
    difficulty = (clean_probability - degraded_probability).abs()
    consistency = (clean_logits - degraded_logits).square()
    return ((1.0 + alpha * difficulty) * consistency).mean()
