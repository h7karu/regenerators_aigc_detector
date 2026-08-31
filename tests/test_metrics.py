import numpy as np
import pytest

from metrics import compute_robustness_metrics


def test_robustness_metrics_average_conditions_and_share_threshold() -> None:
    labels = np.array([0, 0, 1, 1])
    summary = compute_robustness_metrics(
        {
            "clean": (labels, np.array([0.05, 0.10, 0.90, 0.95])),
            "jpeg_50": (labels, np.array([0.10, 0.80, 0.70, 0.90])),
        }
    )

    assert summary["selection_metric"] == "mean_transform_auroc"
    assert summary["selection_score"] == pytest.approx(0.875)
    assert summary["worst_auroc"] == pytest.approx(0.75)
    assert set(summary["transforms"]) == {"clean", "jpeg_50"}
    assert all(
        metrics["threshold"] == summary["threshold"]
        for metrics in summary["transforms"].values()
    )


def test_robustness_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="At least one"):
        compute_robustness_metrics({})
