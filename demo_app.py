"""Shared Gradio interface for local and Colab AIGC-detector demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from demo_inference import DemoDetector


APP_CSS = """
.gradio-container { max-width: 1180px !important; }
.hero { text-align: center; margin: 0.6rem auto 1.4rem auto; }
.hero h1 { margin-bottom: 0.2rem; }
.prototype-note { color: #64748b; font-size: 0.95rem; }
.verdict-card {
  border-radius: 14px;
  padding: 1.15rem 1.3rem;
  border: 2px solid;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
}
.verdict-card.authentic {
  background: #ecfdf5;
  border-color: #34d399;
  color: #14532d !important;
}
.verdict-card.generated {
  background: #fef2f2;
  border-color: #f87171;
  color: #7f1d1d !important;
}
.verdict-card h2 {
  color: inherit !important;
  font-size: 1.8rem !important;
  font-weight: 800 !important;
  line-height: 1.25 !important;
  letter-spacing: -0.015em;
  margin: 0 0 0.7rem 0 !important;
}
.verdict-card .score-line {
  color: #0f172a !important;
  font-size: 1.08rem !important;
  line-height: 1.6 !important;
  margin-bottom: 0.55rem !important;
}
.verdict-card .explanation {
  color: #334155 !important;
  font-size: 0.98rem !important;
  line-height: 1.55 !important;
  margin: 0 !important;
}
.metric-box {
  border: 1px solid #94a3b8 !important;
  border-radius: 10px !important;
  background: #f8fafc !important;
}
.metric-box label,
.metric-box label span {
  color: #334155 !important;
  font-weight: 700 !important;
}
.metric-box input {
  color: #0f172a !important;
  background: #ffffff !important;
  border: 1px solid #cbd5e1 !important;
  font-size: 1.2rem !important;
  font-weight: 800 !important;
  opacity: 1 !important;
  -webkit-text-fill-color: #0f172a !important;
}
"""
DEMO_THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")


def _verdict_markdown(result: dict[str, object]) -> str:
    generated = float(result["score"]) >= float(result["threshold"])
    icon = "🔴" if generated else "🟢"
    state_class = "generated" if generated else "authentic"
    return (
        f"<div class='verdict-card {state_class}'>"
        f"<h2>{icon} {result['verdict']}</h2>"
        f"<p class='score-line'>Model score: "
        f"<strong>{float(result['score']):.4f}</strong> &nbsp;&middot;&nbsp; "
        f"Decision threshold: <strong>{float(result['threshold']):.4f}</strong></p>"
        "<p class='explanation'>The score is not a calibrated probability. "
        "Treat this as experimental forensic evidence, not proof of provenance.</p></div>"
    )


def _robustness_plot(rows: list[dict[str, object]], threshold: float):
    labels = [str(row["condition"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    colours = ["#ef4444" if score >= threshold else "#10b981" for score in scores]
    figure, axis = plt.subplots(figsize=(8.5, 4.0))
    bars = axis.bar(labels, scores, color=colours)
    axis.axhline(
        threshold,
        color="#0f172a",
        linestyle="--",
        linewidth=1.4,
        label=f"threshold = {threshold:.3f}",
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("AI-generated/manipulated model score")
    axis.set_title("Prediction stability under common transformations")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower right")
    axis.bar_label(bars, labels=[f"{score:.3f}" for score in scores], padding=3)
    figure.tight_layout()
    return figure


def create_demo(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    device: str = "auto",
    use_tta: bool = True,
) -> gr.Blocks:
    """Create the complete app; the checkpoint is loaded exactly once."""

    detector = DemoDetector(
        checkpoint_path,
        config_path,
        device=device,
        use_tta=use_tta,
    )
    model_metadata = detector.model_metadata()

    def analyse(image: Image.Image | None):
        try:
            result = detector.predict(image).to_dict()
        except Exception as error:
            raise gr.Error(str(error)) from error
        details = {
            "input": {
                "width": result["width"],
                "height": result["height"],
                "format": result["image_format"],
            },
            "inference": {
                "runtime_ms": round(float(result["runtime_ms"]), 1),
                "device": model_metadata["device"],
            },
            "model": model_metadata,
        }
        return (
            _verdict_markdown(result),
            float(result["score"]),
            float(result["threshold"]),
            round(float(result["runtime_ms"]), 1),
            details,
        )

    def analyse_robustness(image: Image.Image | None):
        try:
            result = detector.analyse_robustness(image)
        except Exception as error:
            raise gr.Error(str(error)) from error
        rows = list(result["rows"])
        summary = (
            "✅ Verdict remained stable across all five conditions."
            if result["all_verdicts_agree"]
            else "⚠️ Verdict changed under at least one transformation."
        )
        summary += (
            f" Score range: {float(result['score_range']):.4f}; "
            f"total runtime: {float(result['runtime_ms']):.0f} ms."
        )
        table = pd.DataFrame(rows).rename(
            columns={
                "condition": "Condition",
                "score": "Model score",
                "verdict": "Verdict",
                "distance_from_threshold": "Distance from threshold",
            }
        )
        for column in ("Model score", "Distance from threshold"):
            table[column] = table[column].map(lambda value: round(float(value), 4))
        return (
            summary,
            _robustness_plot(rows, detector.threshold),
            table,
            result["previews"],
        )

    with gr.Blocks(
        title="Regenerators AI Image Detector",
    ) as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>Regenerators AI Image Detector</h1>
              <p>RGB + Fourier-phase image forensics with compression-robust training</p>
              <p class="prototype-note">Experimental prototype · CIFAKE-initialized and fine-tuned on the 40,000-image SID training split</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                image_input = gr.Image(
                    label="Image to analyse",
                    type="pil",
                    image_mode="RGB",
                    sources=["upload", "webcam", "clipboard"],
                    height=430,
                )
                with gr.Row():
                    analyse_button = gr.Button(
                        "Analyse image", variant="primary", interactive=False
                    )
                    clear_button = gr.ClearButton(value="Clear")

            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab("Detector"):
                        verdict = gr.Markdown(
                            "Upload an image and select **Analyse image**."
                        )
                        with gr.Row():
                            score = gr.Number(
                                label="Model score",
                                precision=4,
                                elem_classes=["metric-box"],
                            )
                            threshold = gr.Number(
                                label="Decision threshold",
                                precision=4,
                                elem_classes=["metric-box"],
                            )
                            runtime = gr.Number(
                                label="Runtime (ms)",
                                precision=1,
                                elem_classes=["metric-box"],
                            )
                        details = gr.JSON(label="Technical details", open=False)

                    with gr.Tab("Robustness Lab"):
                        gr.Markdown(
                            "Compare the prediction after JPEG compression, blur, "
                            "and downscaling. This does not use the reserved test set."
                        )
                        robustness_button = gr.Button(
                            "Run robustness analysis",
                            variant="secondary",
                            interactive=False,
                        )
                        robustness_summary = gr.Markdown()
                        robustness_plot = gr.Plot(label="Score comparison")
                        robustness_table = gr.Dataframe(
                            label="Detailed scores",
                            interactive=False,
                            wrap=True,
                        )
                        preview_gallery = gr.Gallery(
                            label="Transformed previews",
                            columns=5,
                            rows=1,
                            height=210,
                            object_fit="contain",
                        )

        with gr.Accordion("Model scope and limitations", open=False):
            gr.Markdown(
                f"""
- **Checkpoint:** `{model_metadata['checkpoint']}`
- **Configuration:** `{model_metadata['config']}`
- **Selected epoch:** {int(model_metadata['selected_epoch'])}
- **Inference policy:** {model_metadata['inference_policy']}
- **Inference views:** {', '.join(model_metadata['inference_transforms'])}
- **Architecture:** Swin-Tiny RGB branch + Fourier-phase CNN with learned fusion
- **Parameters:** {int(model_metadata['parameters']):,}
- **Pooled validation AUROC:** {float(model_metadata.get('validation_auroc') or 0.0):.4f}
- **Worst-condition validation AUROC:** {float(model_metadata.get('validation_worst_auroc') or 0.0):.4f}
- **Current scope:** fine-tuned on SID real, fully synthetic, and tampered images; broader real-world and unseen-generator generalisation is not established.
- **Interpretation:** pixel-only detection is supporting evidence, not definitive provenance.
                """
            )

        image_input.change(
            fn=lambda image: (
                gr.update(interactive=image is not None),
                gr.update(interactive=image is not None),
            ),
            inputs=image_input,
            outputs=[analyse_button, robustness_button],
            queue=False,
            show_progress="hidden",
        )
        analyse_button.click(
            fn=analyse,
            inputs=image_input,
            outputs=[verdict, score, threshold, runtime, details],
            concurrency_limit=1,
            api_name="analyse",
        )
        robustness_button.click(
            fn=analyse_robustness,
            inputs=image_input,
            outputs=[
                robustness_summary,
                robustness_plot,
                robustness_table,
                preview_gallery,
            ],
            concurrency_limit=1,
            api_name="robustness",
        )
        clear_button.add(
            [
                image_input,
                verdict,
                score,
                threshold,
                runtime,
                details,
                robustness_summary,
                robustness_plot,
                robustness_table,
                preview_gallery,
                analyse_button,
                robustness_button,
            ]
        )

    return demo
