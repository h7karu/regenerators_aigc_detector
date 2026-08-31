import numpy as np
import pytest
import torch

from inference_policy import (
    TTA_TRANSFORMS,
    aggregate_numpy_logits,
    aggregate_tensor_logits,
    logit_threshold,
    standardized_evidence,
)


def test_median_and_trimmed_mean_reject_an_outlier() -> None:
    values = {
        name: np.array([value, value + 1.0])
        for name, value in zip(TTA_TRANSFORMS, (0.0, 1.0, 2.0, 3.0, 100.0))
    }

    median = aggregate_numpy_logits(values, method="median")
    trimmed = aggregate_numpy_logits(values, method="trimmed_mean")

    assert median.tolist() == [2.0, 3.0]
    assert trimmed.tolist() == pytest.approx([2.0, 3.0])


def test_tensor_aggregation_matches_numpy() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0, 100.0])

    assert aggregate_tensor_logits(logits, method="median").item() == 2.0
    assert aggregate_tensor_logits(logits, method="trimmed_mean").item() == 2.0

    batched = torch.stack([logits, logits + 1.0])
    assert aggregate_tensor_logits(batched, method="median").tolist() == [2.0, 3.0]
    assert aggregate_tensor_logits(batched, method="trimmed_mean").tolist() == [
        2.0,
        3.0,
    ]


def test_standardized_evidence_centers_the_threshold() -> None:
    boundary = logit_threshold(0.8)
    evidence = standardized_evidence(
        np.array([boundary - 2.0, boundary, boundary + 2.0]),
        threshold=0.8,
        scale=2.0,
    )

    assert evidence.tolist() == pytest.approx([-1.0, 0.0, 1.0])
