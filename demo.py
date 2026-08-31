"""Launch the Regenerators detector demo locally or with an optional share link."""

from __future__ import annotations

import argparse

from demo_app import APP_CSS, DEMO_THEME, create_demo
from deployment import DEFAULT_MODEL_CHECKPOINT, DEFAULT_MODEL_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_MODEL_CHECKPOINT,
        help=f"Model checkpoint (default: {DEFAULT_MODEL_CHECKPOINT}).",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_MODEL_CONFIG,
        help=f"Model configuration (default: {DEFAULT_MODEL_CONFIG}).",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument(
        "--single-view",
        action="store_true",
        help="Disable the deployed five-view TTA policy.",
    )
    args = parser.parse_args()

    demo = create_demo(
        args.checkpoint,
        args.config,
        device=args.device,
        use_tta=not args.single_view,
    )
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
