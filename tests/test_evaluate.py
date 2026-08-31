import pytest

from evaluate import summarize_transforms


def test_transform_summary_reports_worst_case_and_clean_drop() -> None:
    summary = summarize_transforms(
        {
            "clean": {"auroc": 0.95, "balanced_accuracy": 0.90},
            "jpeg_50": {"auroc": 0.90, "balanced_accuracy": 0.85},
            "blur_1.0": {"auroc": 0.80, "balanced_accuracy": 0.75},
        }
    )

    assert summary["mean_auroc"] == pytest.approx(0.8833333333)
    assert summary["mean_transformed_auroc"] == pytest.approx(0.85)
    assert summary["worst_auroc"] == pytest.approx(0.80)
    assert summary["worst_auroc_transform"] == "blur_1.0"
    assert summary["clean_to_worst_auroc_drop"] == pytest.approx(0.15)
