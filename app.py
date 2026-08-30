"""Local Gradio interface for the AIGC image detector."""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from aigc_detector.ui import DetectorService

ROOT = Path(__file__).resolve().parent
CHECKPOINT = Path(
    os.environ.get(
        "AIGC_CHECKPOINT",
        ROOT / "models" / "notebook_dual_branch.joblib",
    )
)
service = DetectorService(CHECKPOINT)


def analyze(image):
    try:
        return service.predict(image)
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc
    except Exception as exc:
        raise gr.Error(
            "Detection failed. Confirm that the checkpoint and cached CLIP model are available."
        ) from exc


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="AI-Generated Image Detector") as demo:
        gr.Markdown(
            """
            # AI-Generated Image Detector
            Upload an image or take a webcam photo to estimate whether it is
            authentic or AI-generated.
            """
        )

        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    type="pil",
                    sources=["upload", "webcam", "clipboard"],
                    label="Image",
                    height=430,
                )
                with gr.Row():
                    analyze_button = gr.Button("Analyze image", variant="primary")
                    clear_button = gr.Button("Clear", variant="secondary")

            with gr.Column():
                verdict = gr.Markdown("## Waiting for an image")
                probabilities = gr.Label(
                    label="Confidence",
                    num_top_classes=2,
                )

        gr.Markdown(
            """
            **Important:** This demo uses a small experimental checkpoint trained
            on CIFAKE. Results can be wrong, especially for edited, compressed,
            or out-of-distribution images.
            """
        )

        analyze_button.click(
            fn=analyze,
            inputs=image,
            outputs=[probabilities, verdict],
            show_progress="full",
        )
        clear_button.click(
            fn=lambda: (None, None, "## Waiting for an image"),
            outputs=[image, probabilities, verdict],
            queue=False,
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
    )
