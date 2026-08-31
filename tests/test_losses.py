import torch

from losses import difficulty_aware_consistency_loss


def test_consistency_loss_is_zero_for_identical_logits() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0])
    loss = difficulty_aware_consistency_loss(logits, logits)
    assert loss.item() == 0.0


def test_consistency_loss_is_positive_for_disagreement() -> None:
    clean = torch.tensor([-2.0, 2.0])
    degraded = torch.tensor([2.0, -2.0])
    loss = difficulty_aware_consistency_loss(clean, degraded)
    assert loss.item() > 0.0
