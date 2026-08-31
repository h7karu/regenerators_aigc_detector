"""Shared Gradio interface for local and Colab AIGC-detector demos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from demo_inference import DemoDetector


# Every rule below sets foreground and background together, and each token is
# redefined for Gradio's dark class. Setting only one half of the pair is what
# previously produced slate labels on dark panels.
APP_CSS = """
.gradio-container {
  max-width: 1180px !important;
  --rg-surface: #ffffff;
  --rg-raised: #f8fafc;
  --rg-border: #e2e8f0;
  --rg-text: #0f172a;
  --rg-muted: #52627a;
  --rg-real-bg: #f0fdfa;
  --rg-real-edge: #0d9488;
  --rg-real-text: #115e59;
  --rg-fake-bg: #fef5f5;
  --rg-fake-edge: #dc2626;
  --rg-fake-text: #991b1b;
}
.dark .gradio-container,
.gradio-container.dark,
.dark {
  --rg-surface: #101827;
  --rg-raised: #172033;
  --rg-border: #2b3a52;
  --rg-text: #e8eef8;
  --rg-muted: #a3b2c9;
  --rg-real-bg: #0c2521;
  --rg-real-edge: #2dd4bf;
  --rg-real-text: #99f6e4;
  --rg-fake-bg: #2b1417;
  --rg-fake-edge: #f87171;
  --rg-fake-text: #fecaca;
}

.hero {
  margin: 0.4rem auto 1.5rem auto;
  padding-bottom: 1.1rem;
  border-bottom: 1px solid var(--rg-border);
}
.hero h1 {
  color: var(--rg-text) !important;
  font-size: 1.75rem !important;
  font-weight: 650 !important;
  letter-spacing: -0.02em;
  margin: 0 0 0.35rem 0 !important;
}
.hero p { margin: 0.15rem 0 !important; }
.hero .tagline {
  color: var(--rg-muted) !important;
  font-size: 1rem !important;
}
.hero .prototype-note {
  color: var(--rg-muted) !important;
  font-size: 0.86rem !important;
  opacity: 0.85;
}

.verdict-card {
  background: var(--rg-surface);
  border: 1px solid var(--rg-border);
  border-left: 5px solid var(--rg-border);
  border-radius: 10px;
  padding: 1.1rem 1.25rem;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, sans-serif;
}
.verdict-card.authentic {
  background: var(--rg-real-bg);
  border-color: var(--rg-border);
  border-left-color: var(--rg-real-edge);
}
.verdict-card.generated {
  background: var(--rg-fake-bg);
  border-color: var(--rg-border);
  border-left-color: var(--rg-fake-edge);
}
.verdict-card .eyebrow {
  display: block;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--rg-muted) !important;
  margin: 0 0 0.35rem 0 !important;
}
.verdict-card h2 {
  font-size: 1.5rem !important;
  font-weight: 650 !important;
  line-height: 1.25 !important;
  letter-spacing: -0.015em;
  margin: 0 0 0.75rem 0 !important;
}
.verdict-card.authentic h2 { color: var(--rg-real-text) !important; }
.verdict-card.generated h2 { color: var(--rg-fake-text) !important; }
.verdict-card .score-line {
  color: var(--rg-text) !important;
  font-size: 1rem !important;
  line-height: 1.6 !important;
  margin: 0 0 0.5rem 0 !important;
}
.verdict-card .score-line strong {
  font-variant-numeric: tabular-nums;
  font-weight: 650 !important;
}
.verdict-card .explanation {
  color: var(--rg-muted) !important;
  font-size: 0.9rem !important;
  line-height: 1.55 !important;
  margin: 0 !important;
}

.metric-box {
  background: var(--rg-raised) !important;
  border: 1px solid var(--rg-border) !important;
  border-radius: 8px !important;
}
.metric-box label,
.metric-box label span,
.metric-box span[data-testid="block-info"] {
  background: transparent !important;
  color: var(--rg-muted) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
.metric-box input {
  background: transparent !important;
  color: var(--rg-text) !important;
  -webkit-text-fill-color: var(--rg-text) !important;
  border: none !important;
  box-shadow: none !important;
  font-size: 1.3rem !important;
  font-weight: 600 !important;
  font-variant-numeric: tabular-nums;
  opacity: 1 !important;
}
"""
DEMO_THEME = gr.themes.Soft(primary_hue="slate", secondary_hue="slate")


def _verdict_markdown(result: dict[str, object]) -> str:
    generated = float(result["score"]) >= float(result["threshold"])
    state_class = "generated" if generated else "authentic"
    return (
        f"<div class='verdict-card {state_class}'>"
        "<span class='eyebrow'>Assessment</span>"
        f"<h2>{result['verdict']}</h2>"
        f"<p class='score-line'>Score <strong>{float(result['score']):.4f}</strong> "
        f"against a threshold of <strong>{float(result['threshold']):.4f}</strong></p>"
        "<p class='explanation'>Scores are not calibrated probabilities. Use this "
        "as one piece of evidence about an image, not as proof of how it was made."
        "</p>"
        "<p class='explanation'>The 0.4856 threshold was selected on the SID "
        "validation set (2,000 held-out images) after five-view TTA (clean, "
        "JPEG-70, blur, resize, crop) with trimmed-mean logit aggregation, by "
        "picking the point on the ROC curve that maximizes TPR &minus; FPR "
        "(Youden's J). It is not a fixed 0.5 cutoff &mdash; it's the "
        "operating point that best separated real from AI-generated images on "
        "that validation data.</p></div>"
    )


def _robustness_plot(rows: list[dict[str, object]], threshold: float):
    labels = [str(row["condition"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    colours = ["#dc2626" if score >= threshold else "#0d9488" for score in scores]
    figure, axis = plt.subplots(figsize=(8.5, 4.0), facecolor="white")
    axis.set_facecolor("white")
    bars = axis.bar(labels, scores, color=colours, width=0.62)
    axis.axhline(
        threshold,
        color="#475569",
        linestyle="--",
        linewidth=1.2,
        label=f"threshold {threshold:.3f}",
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Model score", fontsize=10, color="#334155")
    axis.set_title(
        "Score under each transformation",
        fontsize=12,
        color="#0f172a",
        pad=12,
        loc="left",
    )
    axis.grid(axis="y", alpha=0.18, color="#94a3b8")
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#cbd5e1")
    axis.tick_params(colors="#475569", labelsize=9)
    legend = axis.legend(loc="lower right", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color("#475569")
    axis.bar_label(
        bars,
        labels=[f"{score:.3f}" for score in scores],
        padding=3,
        fontsize=9,
        color="#334155",
    )
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

    def reset_for_new_image(image: Image.Image | None):
        """Drop the previous result so a new upload never shows stale output.

        Only `interactive` is toggled on the two buttons. Routing a Button
        through anything that also sets `value` blanks its label, because a
        Button's value is its caption.
        """
        ready = image is not None
        return (
            gr.update(interactive=ready),
            gr.update(interactive=ready),
            "Select **Analyse image** to run the detector."
            if ready
            else "Upload an image to run the detector.",
            None,
            None,
            None,
            None,
            "",
            None,
            None,
            None,
        )

    def analyse_robustness(image: Image.Image | None):
        try:
            result = detector.analyse_robustness(image)
        except Exception as error:
            raise gr.Error(str(error)) from error
        rows = list(result["rows"])
        summary = (
            "The verdict held across all five conditions."
            if result["all_verdicts_agree"]
            else "The verdict changed under at least one transformation."
        )
        summary += (
            f" Scores spanned {float(result['score_range']):.4f}, "
            f"measured in {float(result['runtime_ms']):.0f} ms."
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
              <p class="tagline">Image forensics from an RGB backbone and a Fourier-phase branch, trained to survive compression.</p>
              <p class="prototype-note">Research prototype. Initialised on CIFAKE, then fine-tuned on 40,000 SID images.</p>
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
                analyse_button = gr.Button(
                    "Analyse image", variant="primary", interactive=False
                )

            with gr.Column(scale=6):
                with gr.Tabs():
                    with gr.Tab("Detector"):
                        verdict = gr.Markdown(
                            "Upload an image to run the detector."
                        )
                        with gr.Row():
                            score = gr.Number(
                                label="Model score",
                                precision=4,
                                elem_classes=["metric-box"],
                            )
                            threshold = gr.Number(
                                label="Threshold",
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
                            "Re-scores the same image after JPEG compression, blur, "
                            "and downscaling, so you can see how far the verdict "
                            "depends on image quality. This runs on your upload "
                            "only; the reserved test set is never touched."
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

        with gr.Accordion("Model details and limitations", open=False):
            gr.Markdown(
                f"""
- **Checkpoint:** `{model_metadata['checkpoint']}`
- **Configuration:** `{model_metadata['config']}`
- **Selected epoch:** {int(model_metadata['selected_epoch'])}
- **Inference policy:** {model_metadata['inference_policy']}
- **Inference views:** {', '.join(model_metadata['inference_transforms'])}
- **Architecture:** Swin-Tiny RGB branch and Fourier-phase CNN with learned fusion
- **Parameters:** {int(model_metadata['parameters']):,}
- **Pooled validation AUROC:** {float(model_metadata.get('validation_auroc') or 0.0):.4f}
- **Worst-condition validation AUROC:** {float(model_metadata.get('validation_worst_auroc') or 0.0):.4f}
- **Scope:** fine-tuned on SID real, fully synthetic, and tampered images. Behaviour on unseen generators, screenshots, and social-media processing is untested.
- **Reading the score:** pixel-level detection supports a judgement about an image. It does not establish where that image came from.
                """
            )

        image_input.change(
            fn=reset_for_new_image,
            inputs=image_input,
            outputs=[
                analyse_button,
                robustness_button,
                verdict,
                score,
                threshold,
                runtime,
                details,
                robustness_summary,
                robustness_plot,
                robustness_table,
                preview_gallery,
            ],
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

    return demo
