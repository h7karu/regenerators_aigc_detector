"""Launch the Regenerators detector demo locally or with an optional share link."""

from __future__ import annotations

import argparse

from demo_app import APP_CSS, DEMO_THEME, create_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/local_phase_experiment/local_phase_best.pt",
    )
    parser.add_argument("--config", default="configs/local_phase_experiment.yaml")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()

    demo = create_demo(args.checkpoint, args.config, device=args.device)
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open_browser,
        show_error=True,
        theme=DEMO_THEME,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
